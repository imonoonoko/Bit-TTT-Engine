//! BitLlamaBlock - Transformer block with TTT + MLP

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::layers::{KVCache, RMSNorm, SwiGLU, TTTLayer};
use crate::model::config::{BitLlamaConfig, ModelArch};

/// Epsilon for RMSNorm
const RMS_NORM_EPS: f64 = 1e-5;

/// Enum to dispatch between TTT and Attention layers
pub enum LayerDispatch {
    TTT(Box<TTTLayer>),
    Attention(Box<crate::layers::BitAttention>),
}

/// Single transformer block: TTT/Attn + MLP with residual connections
pub struct BitLlamaBlock {
    pub norm1: RMSNorm,
    pub core: LayerDispatch,
    pub norm2: RMSNorm,
    pub mlp: SwiGLU,
}

impl BitLlamaBlock {
    pub fn load(
        cfg: &BitLlamaConfig,
        vb: VarBuilder, // Usually "layers.N" or "model.layers.N"
        device: &candle_core::Device,
    ) -> Result<Self> {
        let dim = cfg.hidden_dim;
        let norm1 = RMSNorm::load(dim, RMS_NORM_EPS, vb.pp("norm1").pp("model.norm"), device)
            .or_else(|_| RMSNorm::load(dim, RMS_NORM_EPS, vb.pp("norm1"), device))
            .or_else(|_| RMSNorm::load(dim, RMS_NORM_EPS, vb.pp("input_layernorm"), device))?;

        let norm2 = RMSNorm::load(dim, RMS_NORM_EPS, vb.pp("norm2").pp("model.norm"), device)
            .or_else(|_| RMSNorm::load(dim, RMS_NORM_EPS, vb.pp("norm2"), device))
            .or_else(|_| {
                RMSNorm::load(dim, RMS_NORM_EPS, vb.pp("post_attention_layernorm"), device)
            })?;

        let mlp_dim = cfg.intermediate_dim.unwrap_or(dim * 4);
        let mlp = SwiGLU::load(dim, mlp_dim, vb.pp("mlp"), device)?;

        // Dispatch Layer Loading based on Config
        let core = match cfg.arch {
            ModelArch::TTT => {
                let ttt = TTTLayer::load(dim, cfg.inner_lr, vb.pp("ttt"), device)?;
                LayerDispatch::TTT(Box::new(ttt))
            }
            ModelArch::Llama => {
                let attn = crate::layers::BitAttention::load(
                    dim,
                    cfg.n_heads,
                    cfg.n_kv_heads,
                    cfg.rope_theta,
                    cfg.max_position_embeddings,
                    vb.pp("self_attn"),
                    device,
                )?;
                LayerDispatch::Attention(Box::new(attn))
            }
        };

        Ok(Self {
            norm1,
            core,
            norm2,
            mlp,
        })
    }

    pub fn device(&self) -> &candle_core::Device {
        self.norm1.weight.device()
    }

    pub fn precompute_packed(&mut self) -> Result<()> {
        match &mut self.core {
            LayerDispatch::TTT(t) => t.precompute_packed()?,
            LayerDispatch::Attention(_) => {} // No precompute needed yet
        }
        self.mlp.precompute_packed()?;
        Ok(())
    }

    pub fn forward(
        &self,
        x: &Tensor,
        w_state: &Tensor,
        kv_cache: &mut Option<KVCache>,
        pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        // [Hybrid Guard] Ensure input is on this block's device
        let block_device = self.device();
        let x = if x.device().same_device(block_device) {
            x.clone()
        } else {
            eprintln!(
                "🚀 [Layer {}] Moving tensor {:?} -> {:?}",
                pos,
                x.device(),
                block_device
            );
            let transferred = x.to_device(block_device)?;
            eprintln!(
                "✅ [Layer {}] Moved result: {:?}",
                pos,
                transferred.device()
            );
            transferred
        };

        let residual = &x;
        let x_norm = self.norm1.forward(&x)?;

        let (mixed_out, w_new) = match &self.core {
            LayerDispatch::TTT(t) => {
                // TTT Path: uses w_state, ignores kv_cache/pos
                t.forward_update(w_state, &x_norm)?
            }
            LayerDispatch::Attention(a) => {
                // Attention Path: uses kv_cache/pos, ignores w_state (passthrough)
                let out = a.forward(&x_norm, kv_cache, pos)?;
                (out, w_state.clone())
            }
        };

        // [Hybrid Guard] Ensure mixed output is on same device as residual before adding
        let mixed_out = if mixed_out.device().same_device(residual.device()) {
            mixed_out
        } else {
            mixed_out.to_device(residual.device())?
        };

        let x_mid = (residual + mixed_out)?;
        let residual = &x_mid;
        let x_norm2 = self.norm2.forward(&x_mid)?;
        let mlp_out = self.mlp.forward(&x_norm2)?;

        // [Hybrid Guard] Ensure MLP output is on same device as residual before adding
        let mlp_out = if mlp_out.device().same_device(residual.device()) {
            mlp_out
        } else {
            mlp_out.to_device(residual.device())?
        };

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

        let (mixed_out, w_final) = match &self.core {
            LayerDispatch::TTT(t) => t.forward_chunkwise(w_state, &x_norm, chunk_size)?,
            LayerDispatch::Attention(a) => {
                // Critical: Chunkwise training for Attention not yet supported with KV Cache state
                // Use a temporary cache or stateless mode (causal mask only)
                // For now, assume training = no past cache
                let mut cache = None;
                let out = a.forward(&x_norm, &mut cache, 0)?;
                (out, w_state.clone())
            }
        };

        let x_mid = (residual + mixed_out)?;
        let residual = &x_mid;
        let x_norm2 = self.norm2.forward(&x_mid)?;
        let mlp_out = self.mlp.forward(&x_norm2)?;
        let x_out = (residual + mlp_out)?;

        Ok((x_out, w_final))
    }
}
