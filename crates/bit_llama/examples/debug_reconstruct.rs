use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

fn main() -> Result<()> {
    // 1. モデルロード (Pythonと同じファイル)
    let model_path = "models/TinyLlama-Adaptive-1.1B/model.safetensors";
    let device = Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };

    println!("🔍 Loading Checkpoint: {}", model_path);

    // 2. ターゲット層のロード
    let layer_path = "model.layers.0.mlp.gate_proj";
    let w_packed = vb.get((5632, 512, 3, 4), &format!("{}.weight_packed", layer_path))?;
    let scales = vb.get((3,), &format!("{}.scales", layer_path))?;

    println!("📊 Tensor Info:");
    println!("  w_packed: {:?}", w_packed.shape());
    println!("  scales:   {:?}", scales.to_vec1::<f32>()?);

    // 3. 最初の12個のパック値を表示 (Pythonと比較用)
    let first_block = w_packed
        .narrow(0, 0, 1)? // [1, 512, 3, 4]
        .narrow(1, 0, 1)?; // [1, 1, 3, 4]
    let first_12 = first_block.flatten_all()?.to_vec1::<f32>()?;
    println!(
        "  First 12 packed values (should match Python): {:?}",
        first_12
    );

    // 4. Rustでの「Python式」再構築
    let (out_dim, in_div_4, num_bases, _four) = w_packed.dims4()?;
    let in_dim = in_div_4 * 4;

    println!("\n🔧 Reconstructing with Python-style logic...");
    println!(
        "  Dimensions: out={}, in_div_4={}, bases={}, in={}",
        out_dim, in_div_4, num_bases, in_dim
    );

    let mut w_recon = Tensor::zeros((out_dim, in_dim), DType::F32, &device)?;

    for base in 0..num_bases {
        println!("  Processing Base {}...", base);

        // Python: w_packed[:, :, base, :]
        let base_packed = w_packed.narrow(2, base, 1)?.squeeze(2)?; // [Out, In/4, 4]

        // 値の解釈 (0, 1, 255) -> (0, 1, -1)
        let vec = base_packed.flatten_all()?.to_vec1::<f32>()?;
        let mapped: Vec<f32> = vec
            .iter()
            .map(|&v| {
                if v > 200.0 {
                    -1.0
                }
                // 255 -> -1
                else if v > 0.5 {
                    1.0
                }
                // 1 -> 1
                else {
                    0.0
                } // 0 -> 0
            })
            .collect();

        let base_tensor = Tensor::from_vec(mapped, (out_dim, in_dim), &device)?;

        // 加算: w_recon += base_tensor * scale
        let scale_val = scales.get(base)?.to_scalar::<f32>()?;
        println!("    Scale: {}", scale_val);
        w_recon = (w_recon + (base_tensor * scale_val as f64)?)?;
    }

    // 5. 最初の数値を表示
    println!("\n🔎 Reconstruction Result:");
    let rec_vec = w_recon.flatten_all()?.to_vec1::<f32>()?;
    println!("  First 10 values: {:?}", &rec_vec[..10]);

    // 統計情報
    let min_val = rec_vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = rec_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = rec_vec.iter().sum();
    let mean = sum / rec_vec.len() as f32;
    let variance: f32 =
        rec_vec.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / rec_vec.len() as f32;
    let std = variance.sqrt();

    println!(
        "  Stats: min={:.6}, max={:.6}, mean={:.6}, std={:.6}",
        min_val, max_val, mean, std
    );
    println!("\n✅ Expected std ≈ 0.016 (based on Python original weight std)");

    Ok(())
}
