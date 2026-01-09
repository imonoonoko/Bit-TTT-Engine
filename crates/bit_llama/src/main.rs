//! Bit-TTT All-in-One Binary (`bit-ttt`)
//!
//! Handlers:
//! - No args: Launch GUI
//! - `run`: Launch CLI Chat
//! - `list`: List Models

mod chat;
mod cli_mode;
mod gui_mode;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "bit-ttt", about = "Bit-TTT Ecosystem All-in-One Tool")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Load a model and start chatting (CLI Mode)
    Run {
        /// Path to .bitt file or model directory
        #[arg(required = true)]
        model_path: String,

        /// Sampling Temperature
        #[arg(short, long, default_value_t = 0.8)]
        temperature: f64,

        /// Max Generation Tokens
        #[arg(long, default_value_t = 200)]
        max_tokens: usize,

        /// System Prompt
        #[arg(long, default_value = "You are a helpful AI assistant.")]
        system: String,
    },
    /// List available models
    List,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Run {
            model_path,
            temperature,
            max_tokens,
            system,
        }) => {
            cli_mode::run_chat(&model_path, temperature, max_tokens, &system)?;
        }
        Some(Commands::List) => {
            cli_mode::list_models()?;
        }
        None => {
            // Default to GUI Mode
            println!("🖥️ Launching GUI Mode...");
            gui_mode::run().map_err(|e| anyhow::anyhow!("GUI Error: {}", e))?;
        }
    }

    Ok(())
}
