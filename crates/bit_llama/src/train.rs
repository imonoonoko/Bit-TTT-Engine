//! Train Module - Training pipeline for Bit-Llama
//!
//! This module provides the training infrastructure:
//! - TrainArgs: CLI arguments for training
//! - checkpoint: State saving/loading utilities
//! - training_loop: Main training loop

pub mod args;
pub mod checkpoint;
pub mod training_loop;

// Re-export public API for backward compatibility
pub use args::TrainArgs;
pub use checkpoint::save_training_state;
pub use training_loop::run;
