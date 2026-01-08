use crate::bit_linear::BitLinear;
use ndarray::{Array1, Array2, ArrayView2};

pub struct TTTLayer {
    pub hidden_dim: usize,
    pub inner_lr: f32,
    pub proj_down: BitLinear,
    pub proj_up: BitLinear,
    // Typically the state is per-sequence, but here we might store it if simulating a continuous stream.
    // For TTT, state is usually transient. We'll return it or pass it.
}

impl TTTLayer {
    pub fn new(hidden_dim: usize, inner_lr: f32) -> Self {
        TTTLayer {
            hidden_dim,
            inner_lr,
            proj_down: BitLinear::new(hidden_dim, hidden_dim / 4),
            proj_up: BitLinear::new(hidden_dim / 4, hidden_dim),
        }
    }

    /// Single step update logic
    /// W_state: (D_small, D_small) hidden weights
    /// x_t: (D) input vector
    /// Returns: (Updated W_state, Feature Reconstruction)
    pub fn forward_state_update(
        &self,
        w_state: &Array2<f32>, // Using f32 for state accumulation to be safe for now, can be i32 later
        x_t: &Array1<f32>,
    ) -> (Array2<f32>, Array1<f32>) {
        // 1. Projection
        let mut feat = self.proj_down.forward(x_t); // (D_small)

        // [Optimization] Normalize feat to prevent gradient explosion (Fix B)
        let norm = feat.dot(&feat).sqrt();
        if norm > 1e-6 {
            feat /= norm;
        }

        // 2. Predict Feature from Feature (using W_state)
        // pred_feat = W_state * feat
        // w_state: (D_s, D_s), feat: (D_s)
        let pred_feat = w_state.dot(&feat);

        // 3. Loss = || pred - feat ||^2
        // error = pred - feat
        let error = &pred_feat - &feat;

        // 4. Gradient
        // Grad = error * feat.T  (Outer product)
        // (D_s) * (D_s) -> (D_s, D_s) matrix
        let d_s = feat.len();
        let error_2d = error
            .clone()
            .into_shape((d_s, 1))
            .expect("BUG: error shape mismatch in TTT gradient calculation");
        let feat_2d = feat
            .clone()
            .into_shape((1, d_s))
            .expect("BUG: feat shape mismatch in TTT gradient calculation");
        let grad = error_2d.dot(&feat_2d);

        // 5. Update
        // w_new = w - lr * grad
        let w_new = w_state - (grad * self.inner_lr);

        (w_new, pred_feat)
    }

    pub fn forward_sequence(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let (seq_len, _dim) = x.dim();
        let d_small = self.hidden_dim / 4;

        let mut w_state = Array2::<f32>::zeros((d_small, d_small));
        let mut outputs = Vec::new();

        for t in 0..seq_len {
            // Extract row t
            let x_t = x.row(t).to_owned(); // Clone for simplicity

            // Update & Predict
            let (w_new, pred_feat) = self.forward_state_update(&w_state, &x_t);
            w_state = w_new;

            // Project Up
            let out_t = self.proj_up.forward(&pred_feat);

            // Residual
            outputs.push(&out_t + &x_t);
        }

        // Stack outputs
        // (This part is a bit verbose in ndarray without helper, assuming flattening)
        let mut result = Array2::<f32>::zeros((seq_len, self.hidden_dim));
        for (i, row) in outputs.iter().enumerate() {
            result.row_mut(i).assign(row);
        }
        result
    }
}
