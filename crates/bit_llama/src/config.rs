//! Bit-Llama Training Configuration
//!
//! Handles project configuration, VRAM estimation, and serialization.

use crate::vocab::ModelType;
use chrono;
use eframe::egui;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProjectConfig {
    pub name: String,
    pub created_at: String,
    // Tokenizer Settings
    pub vocab_size: usize,
    #[serde(default)]
    pub model_type: ModelType,
    // Dataset Settings
    pub val_ratio: f32, // Reserved for future use
    // Training Architecture
    pub model_dim: usize,
    pub layers: usize,
    pub context_len: usize,
    #[serde(default)]
    pub n_heads: usize, // Added
    // Training Hyperparameters
    pub batch_size: usize,
    pub steps: usize,
    pub lr: f64,
    pub min_lr: f64,
    pub warmup_steps: usize,
    pub save_interval: usize,
    #[serde(default = "default_accum_steps")]
    pub accum_steps: usize, // Gradient Accumulation Steps
    #[serde(default = "default_profile")]
    pub profile: String,    // "consumer" (8GB) or "server" (24GB+)

    // Universal Parser Settings
    #[serde(default = "default_input_pattern")]
    pub input_pattern: String,
    #[serde(default)]
    pub template: String,
    #[serde(default)]
    pub use_template: bool,
    // Inference Settings
    #[serde(default = "default_temp")]
    pub inference_temp: f64,
    #[serde(default = "default_max_tokens")]
    pub inference_max_tokens: usize,
}

fn default_input_pattern() -> String {
    "data/*".to_string()
}
fn default_temp() -> f64 { 0.8 }
fn default_max_tokens() -> usize { 100 }
fn default_accum_steps() -> usize { 1 }
fn default_profile() -> String { "consumer".to_string() }

impl Default for ProjectConfig {
    fn default() -> Self {
        Self {
            name: "New Project".to_string(),
            created_at: chrono::Local::now().to_string(),
            vocab_size: 8000,
            model_type: ModelType::Unigram,
            val_ratio: 0.05,
            model_dim: 256,
            layers: 4,        // train_llama.rs default
            context_len: 128, // train_llama.rs default
            n_heads: 8,       // Default
            batch_size: 32,
            steps: 10000,
            lr: 1e-3,
            min_lr: 0.0,
            warmup_steps: 500,
            save_interval: 500,
            accum_steps: 1,
            profile: "consumer".to_string(),
            input_pattern: default_input_pattern(),
            template: "".to_string(),
            use_template: false,
            inference_temp: default_temp(),
            inference_max_tokens: default_max_tokens(),
        }
    }
}

impl ProjectConfig {
    pub fn to_bit_llama_config(&self, inner_lr: f64) -> cortex_rust::BitLlamaConfig {
        cortex_rust::BitLlamaConfig {
            vocab_size: self.vocab_size,
            hidden_dim: self.model_dim,
            num_layers: self.layers,
            inner_lr,
            n_gpu_layers: None,
        }
    }

    pub fn estimate_vram_usage(&self) -> (f32, String, egui::Color32) {
        // Approximate calculation
        // Params: Layers * 12 * Dim^2 (Rough Llama approximation)
        let params = (self.layers as f64) * 12.0 * (self.model_dim as f64).powi(2);

        // Memory Factor based on Profile
        // Server (AdamW + FP16): Weights(2) + Grad(2) + Adam(8) + Overhead = ~12-16 bytes
        // Consumer (SFO + BitNet): Weights(1) + Grad(2) + SFO(4) = ~7 bytes
        let bytes_per_param = if self.profile == "server" {
            16.0
        } else {
            7.0 // Conservative estimate for SFO + BitNet
        };

        let model_mem_bytes = params * bytes_per_param;

        // Activations (rough approximation)
        // Consumer profile usually uses Gradient Checkpointing (implied)?
        // Or we just assume MicroBatch=1.
        // We use batch_size as "Logical Batch Size".
        // Real VRAM usage depends on MicroBatch.
        // If profile=consumer, we assume micro_batch = 1.
        // If profile=server, we assume micro_batch = batch_size.

        let micro_batch = if self.profile == "server" {
            self.batch_size as f64
        } else {
            1.0 // Consumer mode always runs micro-batch 1 to save VRAM
        };

        let activation_bytes = micro_batch
            * (self.context_len as f64)
            * (self.model_dim as f64)
            * (self.layers as f64)
            * 12.0; // Factor for Attention/FFN overhead

        // Overhead for Kernels/Workspace
        let overhead_bytes = 512.0 * 1024.0 * 1024.0; // 512MB

        let total_mb = (model_mem_bytes + activation_bytes + overhead_bytes) / (1024.0 * 1024.0);

        let (status, color) = if total_mb < 8000.0 {
            ("Safe (< 8GB)", egui::Color32::GREEN)
        } else if total_mb < 12000.0 {
            ("Moderate (< 12GB)", egui::Color32::from_rgb(255, 165, 0)) // Orange
        } else if total_mb < 24000.0 {
            ("High (Requires 24GB)", egui::Color32::from_rgb(255, 69, 0)) // Red-Orange
        } else {
            ("Critical (> 24GB)", egui::Color32::RED)
        };

        (total_mb as f32, status.to_string(), color)
    }
}
