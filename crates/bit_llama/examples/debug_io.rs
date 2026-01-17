use candle_core::{DType, Device, Result};
use candle_nn::VarBuilder;

fn main() -> Result<()> {
    // モデルパス
    let model_path = "models/TinyLlama-Adaptive-1.1B/model.safetensors";
    let device = Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };

    println!("🏥 Checking IO Layers...");

    // 1. LM Head Check (最重要容疑者)
    let head = vb.get((32000, 2048), "lm_head.weight")?;
    let head_vec = head.flatten_all()?.to_vec1::<f32>()?;

    println!("\n💀 LM Head Stats:");
    println!("   Shape: {:?}", head.shape());
    println!("   First 5: {:?}", &head_vec[..5]); // これが Pythonの元モデルと一致するか？
    println!("   Last 5:  {:?}", &head_vec[head_vec.len() - 5..]);

    let head_std = (head.sqr()?.sum_all()?.to_scalar::<f32>()? / head_vec.len() as f32).sqrt();
    println!("   Std Dev: {:.6} (Should be ~0.13)", head_std); // 重みの分散チェック

    // 2. Final Norm Check (共犯の可能性)
    // TinyLlamaは RMSNorm なので weight のみ (biasなし)
    let norm = vb.get((2048,), "model.norm.weight")?;
    let norm_vec = norm.flatten_all()?.to_vec1::<f32>()?;

    println!("\n🛡️ Final Norm Stats:");
    println!("   Shape: {:?}", norm.shape());
    println!("   First 5: {:?}", &norm_vec[..5]); // 通常は 1.0 に近い値など
    println!(
        "   Mean:    {:.6}",
        norm_vec.iter().sum::<f32>() / norm_vec.len() as f32
    );

    // 3. Token Embed Check (入口)
    let embed = vb.get((32000, 2048), "model.embed_tokens.weight")?;
    let embed_vec = embed.flatten_all()?.to_vec1::<f32>()?;
    println!("\n🚪 Embed Token Stats:");
    println!("   First 5: {:?}", &embed_vec[..5]);

    Ok(())
}
