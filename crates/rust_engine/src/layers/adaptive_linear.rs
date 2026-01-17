//! AdaptiveBitLinear - Optimized Loading with Rayon & LUT

use super::BitLinear;
use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;
use rayon::prelude::*; // 並列処理用

// 🔥 高速化の要: 0-255 のバイト値を 4つのf32値に変換する「カンニングペーパー」
// 計算を一切せず、メモリから値を拾うだけにします。
static UNPACK_LUT: [[f32; 4]; 256] = {
    let mut table = [[0.0; 4]; 256];
    let mut i = 0;
    while i < 256 {
        let byte = i as u8;
        let mut j = 0;
        while j < 4 {
            // 2bit: 00=0, 01=1, 10=-1, 11=0
            let val = (byte >> (j * 2)) & 0b11;
            table[i][j] = match val {
                1 => 1.0,
                2 => -1.0,
                _ => 0.0,
            };
            j += 1;
        }
        i += 1;
    }
    table
};

#[derive(Clone)]
pub struct AdaptiveBitLinear {
    pub legacy_linear: Option<BitLinear>,
    pub reconstructed_weight: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
}

impl AdaptiveBitLinear {
    pub fn load(in_dim: usize, out_dim: usize, vb: VarBuilder, device: &Device) -> Result<Self> {
        // 1. レガシー (BitNet) の確認
        if let Ok(linear) = BitLinear::load(in_dim, out_dim, vb.clone(), device) {
            return Ok(Self {
                legacy_linear: Some(linear),
                reconstructed_weight: None,
                in_features: in_dim,
                out_features: out_dim,
            });
        }

        // 2. Adaptive Format (Bit-TTT) のロード
        for num_bases in 1..=8 {
            if let Ok(scales) = vb.get((num_bases,), "scales") {
                let packed = vb.get((out_dim, in_dim / 4, num_bases), "weight_packed")?;

                // CPUに一度持ってくる
                let packed_cpu = packed.to_device(&Device::Cpu)?;
                let scales_cpu = scales.to_device(&Device::Cpu)?;

                eprintln!(
                    "🚀 [FAST-LOAD] Loading layer: {}x{} (bases={})",
                    in_dim, out_dim, num_bases
                );

                // 生データを取得 (Type agnostic handling)
                let packed_dtype = packed_cpu.dtype();
                let packed_vec = match packed_dtype {
                    candle_core::DType::U8 => packed_cpu.flatten_all()?.to_vec1::<u8>()?,
                    candle_core::DType::F32 => {
                        eprintln!("⚠️ [FAST-LOAD] Converting F32 packed weights to U8 (Legacy Model Format)");
                        // Use Candle's native cast (optimized)
                        packed_cpu
                            .to_dtype(candle_core::DType::U8)?
                            .flatten_all()?
                            .to_vec1::<u8>()?
                    }
                    _ => {
                        candle_core::bail!("Unexpected dtype for weight_packed: {:?}", packed_dtype)
                    }
                };

                let scales_vec = scales_cpu.to_vec1::<f32>()?;

                // 🚀 【ここが高速化の核心】
                // Rayonを使って「行ごと」に並列処理で解凍・再構築する
                let packed_row_stride = (in_dim / 4) * num_bases;

                let rows: Vec<Vec<f32>> = (0..out_dim)
                    .into_par_iter()
                    .map(|row_idx| {
                        let mut row_w = vec![0.0f32; in_dim];
                        let row_start = row_idx * packed_row_stride;

                        for base in 0..num_bases {
                            let scale = scales_vec[base];

                            for col_pack in 0..(in_dim / 4) {
                                // LUTを使って一瞬で値を取得
                                let flat_idx = row_start + (col_pack * num_bases) + base;
                                let byte_val = packed_vec[flat_idx];
                                let vals = UNPACK_LUT[byte_val as usize];

                                // 加算
                                let out_col_base = col_pack * 4;
                                row_w[out_col_base + 0] += vals[0] * scale;
                                row_w[out_col_base + 1] += vals[1] * scale;
                                row_w[out_col_base + 2] += vals[2] * scale;
                                row_w[out_col_base + 3] += vals[3] * scale;
                            }
                        }
                        row_w
                    })
                    .collect();

                // 結合してTensor化
                let final_flat: Vec<f32> = rows.into_iter().flatten().collect();
                let w_recon = Tensor::from_vec(final_flat, (out_dim, in_dim), device)?;

                return Ok(Self {
                    legacy_linear: None,
                    reconstructed_weight: Some(w_recon),
                    in_features: in_dim,
                    out_features: out_dim,
                });
            }
        }

        candle_core::bail!("Failed to load layer: neither legacy nor adaptive weights found")
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(linear) = &self.legacy_linear {
            return linear.forward(x);
        }
        if let Some(w_recon) = &self.reconstructed_weight {
            // 入力次元の調整 [Batch, Seq, In] -> [Batch*Seq, In]
            let (x_flat, original_shape) = if x.rank() == 3 {
                let (b, s, _) = x.dims3()?;
                (x.flatten(0, 1)?, Some((b, s)))
            } else {
                (x.clone(), None)
            };

            // デバイス整合性チェックと移動
            let w = if w_recon.device().same_device(x_flat.device()) {
                w_recon.clone()
            } else {
                // ここで転送ログを出すとうるさいので、必要な時だけにする
                w_recon.to_device(x_flat.device())?
            };

            let result = x_flat.matmul(&w.t()?)?;

            if let Some((b, s)) = original_shape {
                let (_, out_d) = result.dims2()?;
                return result.reshape((b, s, out_d));
            }
            return Ok(result);
        }
        candle_core::bail!("AdaptiveBitLinear: Invalid State")
    }

    pub fn precompute_packed(&mut self) -> Result<()> {
        if let Some(linear) = &mut self.legacy_linear {
            linear.precompute_packed()?;
        }
        Ok(())
    }
}
