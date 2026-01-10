//! BitLinear - 1.58-bit Quantized Linear Layer

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::TensorExt;

/// 1.58-bit quantized linear layer with STE (Straight-Through Estimator)
pub struct BitLinear {
    pub weight: Tensor,
    #[allow(dead_code)]
    pub in_features: usize,
    #[allow(dead_code)]
    pub out_features: usize,
    /// Pre-computed weights for inference (W_quant.T)
    pub inference_params: Option<Tensor>,
}

impl BitLinear {
    pub fn load(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let init = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let weight = vb.get_with_hints((out_dim, in_dim), "weight", init)?;
        Ok(Self {
            weight,
            in_features: in_dim,
            out_features: out_dim,
            inference_params: None,
        })
    }

    pub fn precompute_for_inference(&mut self) -> Result<()> {
        let w = &self.weight;
        let scale = w.abs()?.mean_all()?;
        let w_scaled = (w / scale.to_scalar::<f32>()? as f64)?;
        let w_quant = w_scaled.round()?.clamp(-1.0, 1.0)?;
        let w_quant_t = w_quant.t()?.detach();
        self.inference_params = Some(w_quant_t);
        Ok(())
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Fast path for inference
        if let Some(w_t) = &self.inference_params {
            return x.matmul_robust(w_t);
        }

        let w = &self.weight;
        let scale = w.abs()?.mean_all()?;
        let w_scaled = (w / scale.to_scalar::<f32>()? as f64)?;
        let w_quant = w_scaled.round()?.clamp(-1.0, 1.0)?;

        // STE
        let diff = (w_quant - &w_scaled)?;
        let detached_diff = diff.detach();
        let w_ste = (detached_diff + &w_scaled)?;

        x.matmul_robust(&w_ste.t()?)
    }
}
