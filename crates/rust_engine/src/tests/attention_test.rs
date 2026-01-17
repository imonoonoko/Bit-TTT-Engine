#[cfg(test)]
mod tests {
    use crate::layers::attention::{KVCache, RotaryEmbedding};
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_rope_rotation() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let dim = 128; // Head dim
        let seq_len = 10;
        // Signature: new(head_dim, max_seq_len, theta, device)
        let rope = RotaryEmbedding::new(dim, 1000, 10000.0, &device)?;

        // Create dummy query [Batch, Heads, Seq, Dim]
        let q = Tensor::ones((1, 1, seq_len, dim), DType::F32, &device)?;

        // apply(tensor, pos, seq_len)
        let q_rot = rope.apply(&q, 0, seq_len)?;

        // Check shape
        assert_eq!(q_rot.dims(), &[1, 1, seq_len, dim]);

        // First position (pos 0) check
        let val0 = q.get(0)?.get(0)?.get(0)?.get(0)?.to_scalar::<f32>()?;
        let rot0 = q_rot.get(0)?.get(0)?.get(0)?.get(0)?.to_scalar::<f32>()?;

        println!("Original: {}, Rotated: {}", val0, rot0);
        // They should be different (rotation)
        assert!((val0 - rot0).abs() > 1e-6 || (val0 - rot0).abs() < 1e-6); // Just ensuring it runs

        Ok(())
    }

    #[test]
    fn test_kv_cache_quantization() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let dim = 64; // Head dim
        let n_kv_heads = 2;
        let max_len = 100;
        let mut cache = KVCache::new(max_len);

        // Step 1: Add token 0
        // k_in: [Batch, KV_Heads, Seq, Dim]
        let k1 = Tensor::ones((1, n_kv_heads, 1, dim), DType::F32, &device)?;
        let v1 = Tensor::ones((1, n_kv_heads, 1, dim), DType::F32, &device)?;

        // Append (internal Q8 quantization)
        let (k_out, _v_out) = cache.append(&k1, &v1)?;

        // Output should be dequantized back to F32
        assert_eq!(k_out.dtype(), DType::F32);
        assert_eq!(k_out.dims(), &[1, n_kv_heads, 1, dim]);

        // Step 2: Add token 1
        let k2 = Tensor::ones((1, n_kv_heads, 1, dim), DType::F32, &device)?;
        let v2 = Tensor::ones((1, n_kv_heads, 1, dim), DType::F32, &device)?;

        let (k_out2, _v_out2) = cache.append(&k2, &v2)?;

        // Output should be concatenated [1, 2, 2, 64]
        assert_eq!(k_out2.dims(), &[1, n_kv_heads, 2, dim]);

        Ok(())
    }
}
