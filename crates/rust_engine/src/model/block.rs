//! BitLlamaBlock - Transformer block with TTT + MLP

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::layers::{RMSNorm, SwiGLU, TTTLayer};

/// Epsilon for RMSNorm
const RMS_NORM_EPS: f64 = 1e-5;

/// Single transformer block: TTT + MLP with residual connections
pub struct BitLlamaBlock {
    pub norm1: RMSNorm,
    pub ttt: TTTLayer,
    pub norm2: RMSNorm,
    pub mlp: SwiGLU,
}

impl BitLlamaBlock {
    pub fn load(dim: usize, inner_lr: f64, vb: VarBuilder) -> Result<Self> {
        let norm1 = RMSNorm::load(dim, RMS_NORM_EPS, vb.pp("norm1"))?;
        let ttt = TTTLayer::load(dim, inner_lr, vb.pp("ttt"))?;
        let norm2 = RMSNorm::load(dim, RMS_NORM_EPS, vb.pp("norm2"))?;
        let mlp_dim = dim * 4;
        let mlp = SwiGLU::load(dim, mlp_dim, vb.pp("mlp"))?;

        Ok(Self {
            norm1,
            ttt,
            norm2,
            mlp,
        })
    }

    pub fn precompute_for_inference(&mut self) -> Result<()> {
        self.ttt.precompute_for_inference()?;
        self.mlp.precompute_for_inference()?;
        Ok(())
    }

    pub fn forward(&self, x: &Tensor, w_state: &Tensor) -> Result<(Tensor, Tensor)> {
        let residual = x;
        let x_norm = self.norm1.forward(x)?;
        let (ttt_out, w_new) = self.ttt.forward_update(w_state, &x_norm)?;
        let x_mid = (residual + ttt_out)?;

        let residual = &x_mid;
        let x_norm2 = self.norm2.forward(&x_mid)?;
        let mlp_out = self.mlp.forward(&x_norm2)?;
        let x_out = (residual + mlp_out)?;

        Ok((x_out, w_new))
    }

    pub fn forward_chunkwise(
        &self,
        x: &Tensor,
        w_state: &Tensor,
        chunk_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let residual = x;
        let x_norm = self.norm1.forward(x)?;
        let (ttt_out, w_final) = self.ttt.forward_chunkwise(w_state, &x_norm, chunk_size)?;
        let x_mid = (residual + ttt_out)?;

        let residual = &x_mid;
        let x_norm2 = self.norm2.forward(&x_mid)?;
        let mlp_out = self.mlp.forward(&x_norm2)?;
        let x_out = (residual + mlp_out)?;

        Ok((x_out, w_final))
    }
}
