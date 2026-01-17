//! SwiGLU - Gated MLP with SiLU activation

use candle_core::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;

use super::AdaptiveBitLinear;

/// SwiGLU MLP block (Gate, Down, Up projections)
pub struct SwiGLU {
    pub w1: AdaptiveBitLinear, // Gate
    pub w2: AdaptiveBitLinear, // Down
    pub w3: AdaptiveBitLinear, // Up
}

impl SwiGLU {
    pub fn load(
        hidden_dim: usize,
        intermediate_dim: usize,
        vb: VarBuilder,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let w1 = AdaptiveBitLinear::load(hidden_dim, intermediate_dim, vb.pp("gate_proj"), device)?;
        let w2 = AdaptiveBitLinear::load(intermediate_dim, hidden_dim, vb.pp("down_proj"), device)?;
        let w3 = AdaptiveBitLinear::load(hidden_dim, intermediate_dim, vb.pp("up_proj"), device)?;
        Ok(Self { w1, w2, w3 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_gate = self.w1.forward(x)?;
        let x_up = self.w3.forward(x)?;
        let silu_gate = candle_nn::ops::silu(&x_gate)?;
        let hidden = (silu_gate * x_up)?;
        self.w2.forward(&hidden)
    }

    pub fn precompute_packed(&mut self) -> Result<()> {
        self.w1.precompute_packed()?;
        self.w2.precompute_packed()?;
        self.w3.precompute_packed()?;
        Ok(())
    }
}
