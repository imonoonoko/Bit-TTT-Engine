//! Bit-TTT Container Exporter (.bitt)
//! Combines config, tokenizer, and weights into a single optimized file.

use anyhow::Result;
use clap::Parser;
use serde_json::Value;
use std::fs::File;
use std::io::Write;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "models/dummy/config.json")]
    config: String,
    #[arg(long, default_value = "data/TinyStories/tokenizer.json")]
    tokenizer: String,
    #[arg(long, default_value = "model_best.safetensors")]
    model: String,
    #[arg(long, default_value = "bit-llama.bitt")]
    output: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("📦 Packaging into custom format: {}", args.output);

    // 1. データの読み込み
    let config_str = std::fs::read_to_string(&args.config)?;
    let tokenizer_str = std::fs::read_to_string(&args.tokenizer)?;

    // ConfigとTokenizerを1つのJSONオブジェクトにまとめる
    let header_json = serde_json::json!({
        "config": serde_json::from_str::<Value>(&config_str)?,
        "tokenizer": serde_json::from_str::<Value>(&tokenizer_str)?
    });
    let header_bytes = serde_json::to_vec(&header_json)?;
    let header_len = header_bytes.len() as u64;

    // 重みファイルを開く
    let mut model_file = File::open(&args.model)?;
    let model_len = model_file.metadata()?.len();

    // 2. 書き込み開始
    let mut output = File::create(&args.output)?;

    // [Magic: 4 bytes]
    output.write_all(b"BITT")?;

    // [Header Len: 8 bytes] (Little Endian)
    output.write_all(&header_len.to_le_bytes())?;

    // [Header JSON]
    output.write_all(&header_bytes)?;

    // [Binary Body] (copy from safetensors)
    std::io::copy(&mut model_file, &mut output)?;

    println!("✅ Created .bitt file!");
    println!("   Magic: BITT");
    println!("   Header: {} bytes (Config + Tokenizer)", header_len);
    println!("   Body:   {} bytes (Weights)", model_len);

    Ok(())
}
