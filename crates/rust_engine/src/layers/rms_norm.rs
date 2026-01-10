//! RMSNorm - Root Mean Square Layer Normalization

use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;

/// Root Mean Square Normalization layer
pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl RMSNorm {
    pub fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight =
            vb.get_with_hints((dim,), "weight", candle_nn::init::DEFAULT_KAIMING_NORMAL)?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = DType::F32;
        let x_f32 = x.to_dtype(internal_dtype)?;
        let dim = x_f32.rank() - 1;
        let hidden_size = x_f32.dim(dim)?;

        let norm_x = (x_f32.sqr()?.sum_keepdim(dim)? / (hidden_size as f64))?;
        let x_normed = x_f32.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;

        let weight = self
            .weight
            .to_dtype(internal_dtype)?
            .broadcast_as(x_normed.shape())?;
        let result = (x_normed * weight)?;

        result.to_dtype(x_dtype)
    }
}
