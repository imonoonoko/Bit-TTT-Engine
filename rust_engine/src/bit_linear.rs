use ndarray::{Array1, Array2};
use rand::Rng;

pub struct BitLinear {
    pub in_features: usize,
    pub out_features: usize,
    pub weights: Array2<i8>, // Stored as -1, 0, 1
                             // For TTT, we might need a separate mechanism for gradients, but standard BitNet doesn't learn weights in test time?
                             // Ah, TTT layer *does* learn. But BitLinear usually refers to the fixed projection layers.
                             // Let's assume BitLinear is fixed-weight for now, or slowly updating.
}

impl BitLinear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        // Initialize randomly with -1, 0, 1
        // Simplified init: Random trinary
        let mut weights = Array2::<i8>::zeros((in_features, out_features));
        for v in weights.iter_mut() {
            let r: f32 = rng.gen();
            if r < 0.33 {
                *v = -1;
            } else if r < 0.66 {
                *v = 1;
            } else {
                *v = 0;
            }
        }

        BitLinear {
            in_features,
            out_features,
            weights,
        }
    }

    /// Forward pass with quantization
    /// x: Input vector (f32)
    /// Returns: Output vector (f32)
    pub fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
        // 1. Quantize Input to i8 (Range [-127, 127])
        // Simplified scale: max abs value mapping to 127
        let max_val = x.fold(0.0f32, |a, &b| a.max(b.abs()));
        let scale = if max_val > 1e-9 { 127.0 / max_val } else { 1.0 };

        let x_quant: Array1<i8> = x.mapv(|v| (v * scale).round().clamp(-127.0, 127.0) as i8);

        // 2. Integer Matrix Multiplication (i8 * i8 -> i32)
        // Manual loop implementation for clarity/no-blas dependency for now
        let mut y_int = Array1::<i32>::zeros(self.out_features);

        for out_idx in 0..self.out_features {
            let mut sum: i32 = 0;
            for in_idx in 0..self.in_features {
                let w = self.weights[[in_idx, out_idx]] as i32;
                let inp = x_quant[in_idx] as i32;
                sum += w * inp;
            }
            y_int[out_idx] = sum;
        }

        // 3. Dequantize
        // y_float = y_int / scale
        // Note: Weights scale is 1.0 (since they are pure {-1, 0, 1})
        // Real BitNet has a weight scale too, but let's ignore for this prototype loop.

        y_int.mapv(|v| (v as f32) / scale)
    }
}
