// Hide console window in release builds on Windows
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::Result;
use bit_llama::cli::{Cli, Commands};
use bit_llama::{data, evaluate, export, gui, inference, train, vocab};
use clap::Parser;

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Gui) | None => {
            if let Err(e) = gui::run() {
                eprintln!("GUI Error: {}", e);
            }
        }
        Some(Commands::Train(args)) => train::run(args)?,
        Some(Commands::Data(args)) => data::run(args)?,
        Some(Commands::Vocab(args)) => vocab::run(args)?,
        Some(Commands::Export(args)) => export::run(args)?,
        Some(Commands::Inference(args)) => inference::run(args)?,
        Some(Commands::Evaluate(args)) => evaluate::run(args)?,
    }

    Ok(())
}
