// Hide console window in release builds on Windows
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::Result;
use bit_llama::cli::{Cli, Commands};
use bit_llama::{data, evaluate, export, gui, inference, train, vocab};
use clap::Parser;

fn main() -> Result<()> {
    // 1. Setup File Logging
    let file_appender = tracing_appender::rolling::daily("logs", "bit_llama.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    // 2. Setup Console Logging
    let env_filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy();

    // 3. Combine Subscribers
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer()) // Stdout
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false),
        ) // File
        .init();

    // 4. Setup Panic Hook
    std::panic::set_hook(Box::new(|panic_info| {
        let payload = panic_info.payload();
        let msg = if let Some(s) = payload.downcast_ref::<&str>() {
            *s
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.as_str()
        } else {
            "Unknown panic"
        };

        let location = panic_info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_default();
        tracing::error!(target: "panic", "🔥 CRASH detected at {}: {}", location, msg);
        // Also print to stderr just in case console is visible
        eprintln!("🔥 CRASH detected at {}: {}", location, msg);
    }));

    tracing::info!("🚀 Bit-Llama started.");

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
