//! Training Arguments - CLI configuration for training

use clap::Args;

/// Training configuration from command line arguments
#[derive(Args, Debug, Clone)]
pub struct TrainArgs {
    #[arg(long, default_value_t = 256)]
    pub dim: usize,

    #[arg(long, default_value_t = 8)]
    pub layers: usize,

    #[arg(long, default_value_t = 128)]
    pub context_len: usize,

    #[arg(long, default_value_t = 16)]
    pub batch_size: usize,

    #[arg(long, default_value_t = 3e-4)]
    pub lr: f64,

    #[arg(long, default_value_t = 1000)]
    pub steps: usize,

    #[arg(long, default_value_t = 100)]
    pub warmup_steps: usize,

    #[arg(long, default_value_t = 1e-5)]
    pub min_lr: f64,

    #[arg(long, required = true)]
    pub data: String,

    #[arg(long)]
    pub output_dir: Option<String>,

    #[arg(long)]
    pub load: Option<String>,

    #[arg(long, default_value_t = 500)]
    pub save_interval: usize,

    #[arg(long, action)]
    pub benchmark: bool,

    #[arg(long, default_value_t = 1)]
    pub accum: usize,
}
