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

    #[arg(long, default_value_t = 0.8)]
    pub temp: f64,
}

pub fn run(args: InferenceArgs) -> Result<()> {
    println!("--- Bit-Llama Inference ---");
    println!("Loading model from: {}", args.model);

    let mut llama = Llama::load_auto(&args.model).map_err(|e| {
        anyhow::anyhow!("Failed to load model: {}\nEnsure directory contains config.json etc.", e)
    })?;

    llama.model.precompute_for_inference()?;

    println!("✅ Model Loaded!");

    let mut current_temp = args.temp;
    let mut current_max_tokens = args.max_tokens;

    loop {
        // Signal that we are ready for input (machine readable)
        eprintln!("<<READY>>");
        // Also print user prompt for human usage
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

        if prompt == "/reset" {
            llama.reset_state()?;
            println!("🔄 Context/Memory has been reset.");
            continue;
        }

        if prompt.starts_with("/temp ") {
            if let Ok(v) = prompt["/temp ".len()..].parse::<f64>() {
                current_temp = v;
                println!("🌡️ Temperature set to {:.2}", current_temp);
            } else {
                println!("❌ Invalid temperature format.");
            }
            continue;
        }

        if prompt.starts_with("/len ") {
            if let Ok(v) = prompt["/len ".len()..].parse::<usize>() {
                current_max_tokens = v;
                println!("📏 Max length set to {}", current_max_tokens);
            } else {
                println!("❌ Invalid length format.");
            }
            continue;
        }

        println!("[Generating...]");
        let callback = |token: &str| -> anyhow::Result<bool> {
            print!("{}", token);
            io::stdout().flush()?;
            Ok(true)
        };

        match llama.stream_completion(prompt, current_max_tokens, current_temp, callback) {
            Ok(_) => println!(),
            Err(e) => println!("Error: {}", e),
        }
    }

    Ok(())
}
