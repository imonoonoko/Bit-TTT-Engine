use anyhow::Result;
use clap::Parser;
use cortex_rust::Llama;
use std::io::Write;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the model directory (containing config.json, tokenizer.json, model.safetensors)
    #[arg(short, long)]
    model: String,

    /// Prompt to generate text from
    #[arg(short, long)]
    prompt: String,

    /// Maximum tokens to generate
    #[arg(short = 'n', long, default_value_t = 100)]
    max_tokens: usize,

    /// Temperature
    #[arg(short, long, default_value_t = 0.7)]
    temp: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Loading model from: {}", args.model);
    let mut model = Llama::new(&args.model)?;

    println!("Generating (Stream)...");
    print!("{}", args.prompt); // Print prompt first
    std::io::stdout().flush()?;

    let start_time = std::time::Instant::now();
    let mut token_count = 0;

    let _full_text =
        model.stream_completion(&args.prompt, args.max_tokens, args.temp, |token| {
            print!("{}", token);
            std::io::stdout().flush()?;
            token_count += 1;
            Ok(true) // Continue
        })?;

    let duration = start_time.elapsed();
    let tps = token_count as f64 / duration.as_secs_f64();

    println!("\n\n--- Done ---");
    println!(
        "Stats: Generated {} tokens in {:.2}s ({:.2} tok/s)",
        token_count,
        duration.as_secs_f64(),
        tps
    );
    Ok(())
}
