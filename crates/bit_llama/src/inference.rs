use anyhow::Result;
use clap::Args;
use cortex_rust::Llama;
use std::io::{self, Write};

#[derive(Args, Debug, Clone)]
pub struct InferenceArgs {
    #[arg(short, long, default_value = ".")]
    pub model: String,

    #[arg(long, default_value_t = 100)]
    pub max_tokens: usize,
}

pub fn run(args: InferenceArgs) -> Result<()> {
    println!("--- Bit-Llama Inference ---");
    println!("Loading model from: {}", args.model);

    let mut llama = Llama::load_auto(&args.model).map_err(|e| {
        anyhow::anyhow!(
            "Failed to load model: {}\nEnsure directory contains config.json etc.",
            e
        )
    })?;

    llama.model.precompute_for_inference()?;

    println!("✅ Model Loaded!");

    loop {
        print!("\n> ");
        io::stdout().flush()?;
        let mut prompt = String::new();
        io::stdin().read_line(&mut prompt)?;
        let prompt = prompt.trim();
        if prompt.is_empty() {
            continue;
        }
        if prompt == "exit" || prompt == "quit" {
            break;
        }

        println!("[Generating...]");
        let callback = |token: &str| -> anyhow::Result<bool> {
            print!("{}", token);
            io::stdout().flush()?;
            Ok(true)
        };

        match llama.stream_completion(prompt, args.max_tokens, 0.8, callback) {
            Ok(_) => println!(),
            Err(e) => println!("Error: {}", e),
        }
    }

    Ok(())
}
