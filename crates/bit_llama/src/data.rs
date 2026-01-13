use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use std::path::PathBuf;

pub mod clean;
pub mod download;
pub mod preprocess;
pub mod concat;
pub mod sampler;

#[derive(Args, Debug, Clone)]
pub struct DataArgs {
    #[command(subcommand)]
    pub command: DataCommand,
}

#[derive(Subcommand, Debug, Clone)]
pub enum DataCommand {
    /// Preprocess text to u32 tokens (Parallel)
    Preprocess(preprocess::PreprocessArgs),
    /// Download dataset
    Download(DownloadArgs),
    /// Clean text (Normalize)
    Clean(CleanArgs),
}

#[derive(Args, Debug, Clone)]
pub struct DownloadArgs {
    /// Output directory
    #[arg(long, default_value = "data/Wiki40b")]
    pub output_dir: PathBuf,

    /// Target dataset (currently only "wiki40b-ja-sample")
    #[arg(long, default_value = "wiki40b-ja-sample")]
    pub dataset: String,
}

#[derive(Args, Debug, Clone)]
pub struct CleanArgs {
    /// Input file
    #[arg(long)]
    pub input: PathBuf,
    /// Output file
    #[arg(long)]
    pub output: PathBuf,
}

pub fn run(args: DataArgs) -> Result<()> {
    match args.command {
        DataCommand::Preprocess(a) => preprocess::run(a),
        DataCommand::Download(a) => run_download(a),
        DataCommand::Clean(a) => run_clean(a),
    }
}

fn run_download(args: DownloadArgs) -> Result<()> {
    std::fs::create_dir_all(&args.output_dir)?;
    // Call download module
    if args.dataset == "wiki40b-ja-sample" {
        download::download_wiki40b_ja_sample(&args.output_dir)?;
    } else {
        println!("Unknown dataset: {}", args.dataset);
    }
    Ok(())
}

fn run_clean(args: CleanArgs) -> Result<()> {
    let input = std::fs::read_to_string(&args.input).context("Failed to read input")?;
    let cleaned = clean::clean_text(&input);
    std::fs::write(&args.output, cleaned).context("Failed to write output")?;
    println!("✨ Cleaned text saved to {:?}", args.output);
    Ok(())
}
