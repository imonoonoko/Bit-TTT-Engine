// Hide console window in release builds on Windows
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![allow(warnings, rust_2018_idioms, clippy::all, clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::wildcard_imports,
    clippy::default_trait_access,
    clippy::missing_errors_doc, // Allow for now to avoid massive diffs
    clippy::too_many_lines      // Allow for legacy main
)]

use anyhow::Result;
use bit_llama::cli::{Cli, Commands};
use bit_llama::{data, evaluate, export, gui, inference, train, vocab};
use clap::Parser;

fn main() -> Result<()> {
    // 0. Portable Mode: Force CWD to be the executable's directory
    //    This ensures "projects/" is always relative to the .exe, not where it was called from.
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            if let Err(e) = std::env::set_current_dir(exe_dir) {
                eprintln!("Failed to set CWD to exe dir: {e}");
            } else {
                eprintln!("📍 Portable Mode: CWD set to {exe_dir:?}");
            }
        }
    }

    // 0. Ensure logs dir exists (Vital for tracing_appender)
    if let Err(e) = std::fs::create_dir_all("logs") {
        eprintln!("Failed to create logs directory: {e}");
    }

    // 0. Setup Panic Hook (EARLY)
    std::panic::set_hook(Box::new(|panic_info| {
        let payload = panic_info.payload();
        let msg = if let Some(s) = payload.downcast_ref::<&str>() {
            *s
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.as_str()
        } else {
            "Unknown panic"
        };

        let location =
            panic_info.location().map(|l| format!("{}:{}", l.file(), l.line())).unwrap_or_default();

        let error_msg = format!("🔥 CRASH detected at {location}: {msg}");

        // Print to stderr
        eprintln!("{error_msg}");

        // Write to simple crash file
        let _ = std::fs::write("CRASH_REPORT.txt", &error_msg);

        // Try using tracing if initialized (might fail if we crashed inside tracing init)
        // tracing::error!(target: "panic", "{}", error_msg);

        // BLOCK thread so user can read it
        println!("\nPress ENTER to exit...");
        let _ = std::io::stdin().read_line(&mut String::new());
    }));

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

    if let Err(e) = tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer()) // Stdout
        .with(tracing_subscriber::fmt::layer().with_writer(non_blocking).with_ansi(false)) // File
        .try_init()
    {
        eprintln!("Failed to init tracing: {e}");
    }

    tracing::info!("🚀 Bit-Llama started.");

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Gui) | None => {
            if let Err(e) = gui::run() {
                eprintln!("GUI Error: {e}");
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
