use anyhow::Result;
use clap::Args;
use serde_json::Value;
use std::fs::File;
use std::io::Write;

#[derive(Args, Debug, Clone)]
pub struct ExportArgs {
    #[arg(long, default_value = "models/dummy/config.json")]
    pub config: String,
    #[arg(long, default_value = "data/TinyStories/tokenizer.json")]
    pub tokenizer: String,
    #[arg(long, default_value = "model_best.safetensors")]
    pub model: String,
    #[arg(long, default_value = "bit-llama.bitt")]
    pub output: String,
}

pub fn run(args: ExportArgs) -> Result<()> {
    println!("📦 Packaging into custom format: {}", args.output);

    let config_str = std::fs::read_to_string(&args.config)?;
    let tokenizer_str = std::fs::read_to_string(&args.tokenizer)?;

    let header_json = serde_json::json!({
        "config": serde_json::from_str::<Value>(&config_str)?,
        "tokenizer": serde_json::from_str::<Value>(&tokenizer_str)?
    });
    let header_bytes = serde_json::to_vec(&header_json)?;
    let header_len = header_bytes.len() as u64;

    let mut model_file = File::open(&args.model)?;
    let model_len = model_file.metadata()?.len();

    let mut output = File::create(&args.output)?;

    output.write_all(b"BITT")?;
    output.write_all(&header_len.to_le_bytes())?;
    output.write_all(&header_bytes)?;

    std::io::copy(&mut model_file, &mut output)?;

    println!("✅ Created .bitt file!");
    println!("   Magic: BITT");
    println!("   Header: {} bytes (Config + Tokenizer)", header_len);
    println!("   Body:   {} bytes (Weights)", model_len);

    Ok(())
}
