//! Model Module - BitLlama model architecture
//!
//! This module contains the complete model implementation:
//! - BitLlamaBlock: Single transformer block with TTT + MLP
//! - BitLlama: Full model with embedding, layers, and LM head
//! - BitLlamaConfig: Model configuration
//! - Llama: High-level API with tokenizer

pub mod block;
pub mod config;
pub mod llama;

pub use block::BitLlamaBlock;
pub use config::BitLlamaConfig;
pub use llama::{BitLlama, Llama};

// Re-export TTTLayer for backward compatibility alias
pub use crate::layers::TTTLayer;
