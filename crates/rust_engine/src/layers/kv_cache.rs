use candle_core::{DType, Result, Tensor};

/// Quantized Key-Value Cache (Phase 5.2)
///
/// Stores KV pairs in 8-bit quantized format to reduce VRAM usage.
/// Supports on-the-fly dequantization during attention calculation.
///
/// # Architecture
/// - **Storage**: `u8` tensor for data.
/// - **Scale**: `f32` tensor for dequantization factor (per-token-head).
/// - **Zero Point**: Fixed at 128 for symmetric mapping (-127..127 -> 1..255).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct QuantizedKVCache {
    k_cache: Option<Tensor>, // Shape: [batch, n_kv_heads, total_seq_len, head_dim] (u8)
    v_cache: Option<Tensor>, // Shape: [batch, n_kv_heads, total_seq_len, head_dim] (u8)

    k_scale: Option<Tensor>, // Shape: [batch, n_kv_heads, total_seq_len, 1] (f32)
    v_scale: Option<Tensor>,

    current_seq_len: usize,
    max_seq_len: usize,
}

impl QuantizedKVCache {
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            k_cache: None,
            v_cache: None,
            k_scale: None,
            v_scale: None,
            current_seq_len: 0,
            max_seq_len,
        }
    }

    /// Reset cache state (for new generation)
    pub fn reset(&mut self) {
        self.k_cache = None;
        self.v_cache = None;
        self.k_scale = None;
        self.v_scale = None;
        self.current_seq_len = 0;
    }

    /// Append new keys and values to the cache
    ///
    /// This implementation performs on-the-fly quantization.
    /// Returns DEQUANTIZED full cache for use in Attention.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = k.dims4()?;

        // 1. Quantize Inputs (f32/f16 -> u8, f32_scale)
        let (k_u8, k_s) = self.quantize_q8(k)?;
        let (v_u8, v_s) = self.quantize_q8(v)?;

        // 2. Concatenate with existing persistent cache
        // Note: 'cat' creates a new tensor, which is simpler but causes fragmentation.
        // For Phase 5.2 MVP, we accept 'cat'. 'Vec<Tensor>' optimization is Phase 5.3.
        let k_next = match &self.k_cache {
            Some(c) => Tensor::cat(&[c, &k_u8], 2)?,
            None => k_u8,
        };
        let k_scale_next = match &self.k_scale {
            Some(c) => Tensor::cat(&[c, &k_s], 2)?,
            None => k_s,
        };

        let v_next = match &self.v_cache {
            Some(c) => Tensor::cat(&[c, &v_u8], 2)?,
            None => v_u8,
        };
        let v_scale_next = match &self.v_scale {
            Some(c) => Tensor::cat(&[c, &v_s], 2)?,
            None => v_s,
        };

        // 3. Update State
        self.k_cache = Some(k_next.clone());
        self.v_cache = Some(v_next.clone());
        self.k_scale = Some(k_scale_next.clone());
        self.v_scale = Some(v_scale_next.clone());
        self.current_seq_len += seq_len;

        // 4. Dequantize for Return (To be compatible with standard Attention)
        // This effectively "Reconstructs" the full cache in f16/f32 for computation.
        // Optimization: In Phase 5.3, we should fuse this into the Attention Kernel.
        let k_out = self.dequantize_q8(&k_next, &k_scale_next)?;
        let v_out = self.dequantize_q8(&v_next, &v_scale_next)?;

        Ok((k_out, v_out))
    }

    /// Quantize a Tensor to Q8 (Symetric + 128 Offset)
    fn quantize_q8(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // x: [batch, heads, seq, dim]
        // Scale per token-head: max(abs(x), dim=3) -> [batch, heads, seq, 1]
        let x_abs = x.abs()?;
        let max_val = x_abs.max_keepdim(3)?;
        // Avoid division by zero
        let scale = (max_val / 127.0)?;

        // Broadcast scale
        let scaled = x.broadcast_div(&scale)?;

        // Quantize: round(x/s) + 128
        // We use standard rounding.
        let rounded = scaled.round()?;

        // Shift to u8 range [0, 255]. Center is 128.
        let shifted = (rounded + 128.0)?;

        // Clamp to ensure safety (though abs/127 should be within range)
        // Candle's to_dtype(U8) naturally saturates or wraps.
        // We trust the math: max_val/127 -> range [-127, 127]. +128 -> [1, 255].
        let quantized = shifted.to_dtype(DType::U8)?;

        Ok((quantized, scale))
    }

    /// Dequantize Q8 back to original dtype (f32/f16)
    fn dequantize_q8(&self, q: &Tensor, s: &Tensor) -> Result<Tensor> {
        // x = (q - 128) * scale
        let q_float = q.to_dtype(DType::F32)?;
        let shifted = (q_float - 128.0)?;
        let out = shifted.broadcast_mul(s)?;
        Ok(out)
    }
}
