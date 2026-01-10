use crate::data::DataArgs;
use crate::evaluate::EvaluateArgs;
use crate::export::ExportArgs;
use crate::inference::InferenceArgs;
use crate::train::TrainArgs;
use crate::vocab::VocabArgs;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about = "Bit-TTT Unified Toolchain", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Launch the GUI (Default)
    Gui,

    /// Train a model
    Train(TrainArgs),

    /// Preprocess data (Text -> .u32)
    Data(DataArgs),

    /// Train tokenizer (BPE)
    Vocab(VocabArgs),

    /// Export model to .bitt format
    Export(ExportArgs),

    /// Run inference
    Inference(InferenceArgs),

    /// Evaluate model (Perplexity)
    Evaluate(EvaluateArgs),
}
