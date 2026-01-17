//! AdaptiveBitLinear - Multi-Base Quantized Linear Layer
//!
//! Supports:
//! 1. Legacy BitNet (1 base, separate weights) - via backward compatibility
//! 2. Adaptive BitNet (N bases, interleaved packed weights) - "The Fused Path"

#[allow(unused_imports)]
use super::{BitLinear, TensorExt};
use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;

/// Adaptive 1.58-bit Linear Layer
/// Can hold either a single Legacy BitLinear OR a Pre-reconstructed Weight Matrix.
#[derive(Clone)]
pub struct AdaptiveBitLinear {
    /// Legacy Single-Base mode (for backward compat or N=1)
    pub legacy_linear: Option<BitLinear>,

    /// Pre-reconstructed weight matrix [Out, In] (F32)
    /// Computed at load time from packed weights + scales
    pub reconstructed_weight: Option<Tensor>,

    pub in_features: usize,
    pub out_features: usize,
}

impl AdaptiveBitLinear {
    pub fn load(in_dim: usize, out_dim: usize, vb: VarBuilder, device: &Device) -> Result<Self> {
        // 1. Try Loading Adaptive Format (Detect NumBases via scales)
        for nb in 1..=8 {
            if let Ok(scales) = vb.get((nb,), "scales") {
                // Found scales with dimension 'nb'.
                let num_bases = nb;

                // DEBUG info
                eprintln!(
                    "🔥 [ADAPTIVE] Loading layer: in={}, out={}, bases={}, device={:?}",
                    in_dim, out_dim, num_bases, device
                );
                let packed = match vb.get((out_dim, in_dim / 4, num_bases), "weight_packed") {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("❌ Failed to load packed weights: {:?}", e);
                        return Err(e);
                    }
                };

                // --- Pre-compute reconstructed weights at load time ---
                // This uses the verified logic from debug_reconstruct.rs

                let packed_cpu = packed.to_device(&Device::Cpu)?;
                let scales_cpu = scales.to_device(&Device::Cpu)?;

                let mut w_recon =
                    Tensor::zeros((out_dim, in_dim), candle_core::DType::F32, &Device::Cpu)?;

                for base in 0..num_bases {
                    // Python: w_packed[:, :, base, :]
                    // Rust: narrow(2, base, 1) -> squeeze(2) -> [Out, In/4, 4]
                    let base_packed = packed_cpu.narrow(2, base, 1)?.squeeze(2)?;

                    // Unpack 2-bit values
                    // 0 -> 00 -> 0
                    // 1 -> 01 -> 1
                    // 2 -> 10 -> -1
                    // 3 -> 11 -> 0 (padding/unused)
                    let vec = base_packed.flatten_all()?.to_vec1::<f32>()?;
                    let mut mapped: Vec<f32> = Vec::with_capacity(vec.len() * 4);

                    for &v_float in &vec {
                        let v = v_float as u8; // Convert back to u8 (safe since load was U8)

                        for i in 0..4 {
                            let shift = i * 2;
                            let val = (v >> shift) & 0x03;
                            let float_val = match val {
                                1 => 1.0,
                                2 => -1.0,
                                _ => 0.0,
                            };
                            mapped.push(float_val);
                        }
                    }

                    let base_tensor = Tensor::from_vec(mapped, (out_dim, in_dim), &Device::Cpu)?;

                    // w_recon += base_tensor * scale
                    let scale_val = scales_cpu.get(base)?.to_scalar::<f32>()?;
                    w_recon = (w_recon + (base_tensor * scale_val as f64)?)?;
                }

                // Move to target device
                let w_recon = w_recon.to_device(device)?;

                // DEBUG: Print stats for first MLP layer to verify reconstruction
                if out_dim == 5632 && in_dim == 2048 {
                    let w_vec = w_recon
                        .to_device(&Device::Cpu)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let first_10: Vec<f32> = w_vec.iter().take(10).cloned().collect();
                    let sum: f32 = w_vec.iter().sum();
                    let mean = sum / w_vec.len() as f32;
                    let variance: f32 =
                        w_vec.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / w_vec.len() as f32;
                    let std = variance.sqrt();
                    eprintln!(
                        "📊 [VERIFY] gate_proj recon: first10={:?}, std={:.6} (expected ~0.013)",
                        first_10, std
                    );
                }

                return Ok(Self {
                    legacy_linear: None,
                    reconstructed_weight: Some(w_recon),
                    in_features: in_dim,
                    out_features: out_dim,
                });
            }
        }

        // 2. Fallback to Legacy BitLinear
        // This expects "weight" to exist.
        match BitLinear::load(in_dim, out_dim, vb.clone(), device) {
            Ok(linear) => Ok(Self {
                legacy_linear: Some(linear),
                reconstructed_weight: None,
                in_features: in_dim,
                out_features: out_dim,
            }),
            Err(e) => Err(e), // Propagate error if neither found
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Legacy Path
        if let Some(linear) = &self.legacy_linear {
            return linear.forward(x);
        }

        // 2. Adaptive Path (using pre-computed weight matrix)
        if let Some(w_recon) = &self.reconstructed_weight {
            // Handle Rank 3 input: [Batch, Seq, In] -> [Batch*Seq, In]
            let (x_flat, original_shape) = if x.rank() == 3 {
                let (b, s, _) = x.dims3()?;
                (x.flatten(0, 1)?, Some((b, s)))
            } else {
                (x.clone(), None)
            };

            // [Hybrid Guard] Ensure weight is on same device as input
            let w = if w_recon.device().same_device(x_flat.device()) {
                w_recon.clone()
            } else {
                eprintln!(
                    "⚠️ [ADAPTIVE] Moving weight {:?} -> {:?}",
                    w_recon.device(),
                    x_flat.device()
                );
                w_recon.to_device(x_flat.device())?
            };

            // Matmul: [Batch*Seq, In] x [In, Out] = [Batch*Seq, Out]
            let result = x_flat.matmul(&w.t()?)?;

            // Reshape back if needed
            if let Some((b, s)) = original_shape {
                let (_, out_d) = result.dims2()?;
                return result.reshape((b, s, out_d));
            }
            return Ok(result);
        }

        candle_core::bail!("AdaptiveBitLinear: Invalid State (No weights loaded)")
    }

    pub fn precompute_packed(&mut self) -> Result<()> {
        if let Some(linear) = &mut self.legacy_linear {
            linear.precompute_packed()?;
        }
        // Adaptive weights are already reconstructed at load time.
        Ok(())
    }
}
