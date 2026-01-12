//! TTTLayer - Test-Time Training with Online Learning

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::BitLinear;

/// Epsilon for TTT layer normalization
const TTT_NORM_EPS: f32 = 1e-6;

/// Test-Time Training layer with online gradient descent
pub struct TTTLayer {
    #[allow(dead_code)]
    pub hidden_dim: usize,
    #[allow(dead_code)]
    pub d_small: usize,
    pub proj_down: BitLinear,
    pub proj_up: BitLinear,
    pub inner_lr: f64,
}

impl TTTLayer {
    pub fn load(hidden_dim: usize, inner_lr: f64, vb: VarBuilder, device: &candle_core::Device) -> Result<Self> {
        let d_small = hidden_dim / 4;
        Ok(Self {
            hidden_dim,
            d_small,
            proj_down: BitLinear::load(hidden_dim, d_small, vb.pp("down"), device)?,
            proj_up: BitLinear::load(d_small, hidden_dim, vb.pp("up"), device)?,
            inner_lr,
        })
    }

    pub fn precompute_for_inference(&mut self) -> Result<()> {
        self.proj_down.precompute_for_inference()?;
        self.proj_up.precompute_for_inference()?;
        Ok(())
    }

    /// Sequential forward with weight update
    /// w_state: (B, D_small, D_small) or (D_small, D_small)
    /// x: (B, Hidden) or (Hidden)
    pub fn forward_update(&self, w_state: &Tensor, x_t: &Tensor) -> Result<(Tensor, Tensor)> {
        let feat = self.proj_down.forward(x_t)?;

        // Normalize (L2 per vector)
        let last_dim = feat.rank() - 1;
        let norm = feat.sqr()?.sum_keepdim(last_dim)?.sqrt()?;
        let norm = norm.broadcast_add(&Tensor::new(&[TTT_NORM_EPS], x_t.device())?)?;
        let feat_norm = feat.broadcast_div(&norm)?;

        // Predict
        let feat_expanded = feat_norm.unsqueeze(last_dim + 1)?;
        let pred_inner = w_state.matmul(&feat_expanded)?.squeeze(last_dim + 1)?;

        // Loss & Grad
        let diff = (&pred_inner - &feat_norm)?;
        let diff_ed = diff.unsqueeze(last_dim + 1)?;
        let feat_ed_t = feat_norm.unsqueeze(last_dim)?;
        let grad = diff_ed.matmul(&feat_ed_t)?;

        // Update
        let w_new = (w_state - grad * self.inner_lr)?.detach();

        // Project Up
        let out_feat = self.proj_up.forward(&pred_inner)?;

        Ok((out_feat, w_new))
    }

    /// Parallel chunkwise implementation
    /// x: (B, T, Hidden)
    /// w_state: (B, D_small, D_small)
    /// Returns: (output: (B, T, Hidden), w_final: (B, D_small, D_small))
    pub fn forward_chunkwise(
        &self,
        w_state: &Tensor,
        x: &Tensor,
        chunk_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let feat = self.proj_down.forward(x)?;

        // Normalize
        let norm = feat.sqr()?.sum_keepdim(2)?.sqrt()?;
        let norm = norm.broadcast_add(&Tensor::new(&[TTT_NORM_EPS], x.device())?)?;
        let feat_norm = feat.broadcast_div(&norm)?;

        let (_b_sz, t_len, _d_small) = feat_norm.dims3()?;
        let mut current_w = w_state.clone();
        let mut outputs = Vec::new();

        let num_chunks = t_len.div_ceil(chunk_size);

        for i in 0..num_chunks {
            let start = i * chunk_size;
            let len = std::cmp::min(chunk_size, t_len - start);

            let x_chunk = feat_norm.narrow(1, start, len)?;
            let x_chunk_t = x_chunk.transpose(1, 2)?;
            let z_chunk_t = current_w.matmul(&x_chunk_t)?;
            let z_chunk = z_chunk_t.transpose(1, 2)?;
            let diff = (&z_chunk - &x_chunk)?;
            let diff_t = diff.transpose(1, 2)?;
            let grad = diff_t.matmul(&x_chunk)?;

            current_w = (current_w - grad * self.inner_lr)?;
            outputs.push(z_chunk);
        }

        let pred_all = Tensor::cat(&outputs, 1)?;
        let out_feat = self.proj_up.forward(&pred_all)?;

        Ok((out_feat, current_w))
    }
}
