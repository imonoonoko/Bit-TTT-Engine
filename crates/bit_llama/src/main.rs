// Hide console window in release builds on Windows
// #![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::Result;
use bit_llama::cli::{Cli, Commands};
use bit_llama::{data, evaluate, export, gui, inference, train, vocab};
use clap::Parser;

fn main() -> Result<()> {
    // Initialize logging
    let filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy();

    tracing_subscriber::fmt().with_env_filter(filter).init();

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
