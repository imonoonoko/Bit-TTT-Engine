use super::AdaptiveBitLinear;
use candle_core::{Device, Result, Tensor};
use candle_nn::{ops::softmax, VarBuilder};

/// Rotary Position Embedding for TinyLlama
/// Based on the LLaMA/RoPE paper formulation
#[derive(Clone)]
pub struct RotaryEmbedding {
    pub cos_cache: Tensor,
    pub sin_cache: Tensor,
    pub head_dim: usize,
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> Result<Self> {
        // Compute inverse frequencies: 1 / (theta^(2i/dim)) for i in 0..dim/2
        let half_dim = head_dim / 2;
        let mut inv_freq: Vec<f32> = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let freq = 1.0 / (theta.powf((2 * i) as f64 / head_dim as f64)) as f32;
            inv_freq.push(freq);
        }
        let inv_freq = Tensor::from_vec(inv_freq, (1, half_dim), device)?;

        // Compute position indices
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::from_vec(positions, (max_seq_len, 1), device)?;

        // Compute freqs: [max_seq_len, half_dim]
        let freqs = positions.matmul(&inv_freq)?;

        // Compute cos and sin caches
        let cos_cache = freqs.cos()?;
        let sin_cache = freqs.sin()?;

        Ok(Self {
            cos_cache,
            sin_cache,
            head_dim,
        })
    }

    /// Apply rotary embedding to input tensor
    /// Input shape: [batch, heads, seq_len, head_dim]
    pub fn apply(&self, x: &Tensor, pos: usize, seq_len: usize) -> Result<Tensor> {
        let half_dim = self.head_dim / 2;
        let (batch, heads, _, _) = x.dims4()?;

        // Get cos/sin for this sequence length at specific position offset
        // indices: pos .. pos + seq_len
        // BUT our cache is [max_seq, half_dim]
        // We need to slice: [pos .. pos+seq_len]

        let cos = self.cos_cache.narrow(0, pos, seq_len)?; // [seq_len, half_dim]
        let sin = self.sin_cache.narrow(0, pos, seq_len)?;

        // [Hybrid Guard] Ensure RoPE cache is on same device as input
        let (cos, sin) = if cos.device().same_device(x.device()) {
            (cos, sin)
        } else {
            (cos.to_device(x.device())?, sin.to_device(x.device())?)
        };

        // Split x into first half and second half
        let x1 = x.narrow(3, 0, half_dim)?; // [batch, heads, seq, half_dim]
        let x2 = x.narrow(3, half_dim, half_dim)?; // [batch, heads, seq, half_dim]

        // Reshape cos/sin for broadcasting: [1, 1, seq_len, half_dim] -> broadcast to [batch, heads, seq, half_dim]
        let cos = cos
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((batch, heads, seq_len, half_dim))?;
        let sin = sin
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((batch, heads, seq_len, half_dim))?;

        // Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        let out1 = ((&x1 * &cos)? - (&x2 * &sin)?)?;
        let out2 = ((&x1 * &sin)? + (&x2 * &cos)?)?;

        // Concatenate back
        Tensor::cat(&[&out1, &out2], 3)
    }
}

#[derive(Clone)]
pub struct BitAttention {
    pub q_proj: AdaptiveBitLinear,
    pub k_proj: AdaptiveBitLinear,
    pub v_proj: AdaptiveBitLinear,
    pub o_proj: AdaptiveBitLinear,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub scaling: f64,
    pub rotary_emb: RotaryEmbedding,
}

// [Phase 5.2] Use QuantizedKVCache for memory optimization
pub use super::QuantizedKVCache as KVCache;

impl BitAttention {
    pub fn load(
        hidden_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        rope_theta: f64,
        max_position_embeddings: usize,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let head_dim = hidden_dim / n_heads;
        let scaling = 1.0 / (head_dim as f64).sqrt();

        // DEBUG: Print attention params to verify GQA config
        eprintln!(
            "🔍 [ATTN] n_heads={}, n_kv_heads={}, head_dim={} theta={} device={:?}",
            n_heads, n_kv_heads, head_dim, rope_theta, device
        );

        // HF Keys: q_proj, k_proj, v_proj, o_proj
        let q_proj =
            AdaptiveBitLinear::load(hidden_dim, n_heads * head_dim, vb.pp("q_proj"), device)?;
        let k_proj =
            AdaptiveBitLinear::load(hidden_dim, n_kv_heads * head_dim, vb.pp("k_proj"), device)?;
        let v_proj =
            AdaptiveBitLinear::load(hidden_dim, n_kv_heads * head_dim, vb.pp("v_proj"), device)?;
        let o_proj =
            AdaptiveBitLinear::load(n_heads * head_dim, hidden_dim, vb.pp("o_proj"), device)?;

        // RoPE: Use config values (supports Llama-3 theta=500,000)
        let rotary_emb =
            RotaryEmbedding::new(head_dim, max_position_embeddings, rope_theta, device)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_heads,
            n_kv_heads,
            head_dim,
            scaling,
            rotary_emb,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<KVCache>,
        pos: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k_new = self.k_proj.forward(x)?;
        let v_new = self.v_proj.forward(x)?;

        // Shape: [Batch, Seq, Heads * Dim] -> [Batch, Seq, Heads, Dim] -> [Batch, Heads, Seq, Dim]
        let q = q
            .reshape((b_sz, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k_new
            .reshape((b_sz, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let mut v = v_new
            .reshape((b_sz, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // DEBUG: Check devices before RoPE
        // if q.device().is_cuda() != self.rotary_emb.cos.device().is_cuda() {
        //     eprintln!("⚠️ [ATTN] Device Mismatch Pre-RoPE! Q: {:?}, RoPE: {:?}", q.device(), self.rotary_emb.cos.device());
        // }

        // DEBUG: Check devices before RoPE
        if q.device().is_cuda() != self.rotary_emb.cos_cache.device().is_cuda() {
            eprintln!(
                "⚠️ [ATTN] Device Mismatch Pre-RoPE! Q: {:?}, RoPE: {:?}",
                q.device(),
                self.rotary_emb.cos_cache.device()
            );
        }

        // DEBUG: Check devices before RoPE
        if q.device().is_cuda() != self.rotary_emb.cos_cache.device().is_cuda() {
            eprintln!(
                "⚠️ [ATTN] Device Mismatch Pre-RoPE! Q: {:?}, RoPE: {:?}",
                q.device(),
                self.rotary_emb.cos_cache.device()
            );
        }

        // DEBUG: Check devices before RoPE
        if q.device().is_cuda() != self.rotary_emb.cos_cache.device().is_cuda() {
            eprintln!(
                "⚠️ [ATTN] Device Mismatch Pre-RoPE! Q: {:?}, RoPE: {:?}",
                q.device(),
                self.rotary_emb.cos_cache.device()
            );
        }

        // DEBUG: Check devices before RoPE
        if q.device().is_cuda() != self.rotary_emb.cos_cache.device().is_cuda() {
            eprintln!(
                "⚠️ [ATTN] Device Mismatch Pre-RoPE! Q: {:?}, RoPE: {:?}",
                q.device(),
                self.rotary_emb.cos_cache.device()
            );
        }

        // DEBUG: Check devices before RoPE
        if q.device().is_cuda() != self.rotary_emb.cos_cache.device().is_cuda() {
            eprintln!(
                "⚠️ [ATTN] Device Mismatch Pre-RoPE! Q: {:?}, RoPE: {:?}",
                q.device(),
                self.rotary_emb.cos_cache.device()
            );
        }

        // Apply RoPE to new Q and new K
        // q: [batch, heads, seq_len, dim] -> rotated at pos..pos+seq_len
        // k: [batch, kv_heads, seq_len, dim] -> rotated at pos..pos+seq_len

        let q = self.rotary_emb.apply(&q, pos, seq_len)?;
        // Make k mutable for caching concat later
        let mut k = self.rotary_emb.apply(&k, pos, seq_len)?;

        // NOW Update Cache
        // [Phase 5.2] Update Cache (Quantized)
        // Note: cache must be initialized by caller with max_seq_len
        if let Some(cache) = kv_cache {
            let (k_out, v_out) = cache.append(&k, &v)?;
            k = k_out;
            v = v_out;
        }
        // If no cache (e.g. initial prefill without persistent state?), we use k, v as is.

        // GQA handling: Repeat K/V if n_kv_heads < n_heads
        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        // Attn = Softmax(Q @ K.T / sqrt(dim))
        let att = (q.matmul(&k.t()?)? * self.scaling)?;

        // Causal mask
        // K seq_len is now Total Length (cache + new)
        // Q seq_len is New Length
        // We need mask if Q seq_len > 1 (prefill).
        // If generation (Q=1), we attend to everything (no mask needed usually, or mask future if any).
        // Since K is past+present, we attend to all K.

        let (_, _, k_len, _) = k.dims4()?;
        let att = self.apply_causal_mask(&att, seq_len, k_len)?;

        let att = softmax(&att, candle_core::D::Minus1)?;

        // Out = Attn @ V
        let y = att.matmul(&v)?;

        // Reassemble: [Batch, Heads, Seq, Dim] -> [Batch, Seq, Heads, Dim] -> [Batch, Seq, Hidden]
        let y = y.transpose(1, 2)?.reshape((b_sz, seq_len, hidden))?;

        let y = self.o_proj.forward(&y)?;

        Ok(y)
    }

    // GQA handling: Repeat K/V if n_kv_heads < n_heads
    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.n_heads / self.n_kv_heads;
        if n_rep == 1 {
            return Ok(x);
        }
        let (b, n_kv, s, d) = x.dims4()?;

        // Manual repeat: [B, N_KV, S, D] -> [B, N_KV, Rep, S, D] -> [B, N_KV*Rep, S, D]
        x.unsqueeze(2)?
            .expand((b, n_kv, n_rep, s, d))?
            .reshape((b, n_kv * n_rep, s, d))
    }

    fn apply_causal_mask(&self, att: &Tensor, seq_len: usize, k_len: usize) -> Result<Tensor> {
        if seq_len == 1 {
            // Single token generation: attend to all past tokens
            return Ok(att.clone());
        }

        // For prefill (seq_len > 1), we need causal mask.
        // att shape: [batch, heads, seq_len, k_len]
        let past_len = k_len - seq_len;

        // Create mask: 0 if j <= i + past_len, else -inf
        // Standard Llama causal mask
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..k_len).map(move |j| {
                    if j <= i + past_len {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();

        // Create mask on CPU first, then move to checking device
        // This avoids "device mismatch" if att.device() is passed but implementation defaults to Cpu
        let mask = Tensor::from_vec(mask, (1, 1, seq_len, k_len), &Device::Cpu)?
            .to_dtype(att.dtype())?
            .to_device(att.device())?;

        att.broadcast_add(&mask)
    }
}
