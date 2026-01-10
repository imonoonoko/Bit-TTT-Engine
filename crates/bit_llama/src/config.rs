//! Bit-Llama Training Configuration
//!
//! Handles project configuration, VRAM estimation, and serialization.

use chrono;
use eframe::egui;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProjectConfig {
    pub name: String,
    pub created_at: String,
    // Tokenizer Settings
    pub vocab_size: usize,
    // Dataset Settings
    pub val_ratio: f32, // Reserved for future use
    // Training Architecture
    pub model_dim: usize,
    pub layers: usize,
    pub context_len: usize,
    pub n_heads: usize, // Added
    // Training Hyperparameters
    pub batch_size: usize,
    pub steps: usize,
    pub lr: f64,
    pub min_lr: f64,
    pub warmup_steps: usize,
    pub save_interval: usize,
}

impl Default for ProjectConfig {
    fn default() -> Self {
        Self {
            name: "New Project".to_string(),
            created_at: chrono::Local::now().to_string(),
            vocab_size: 8000,
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
        }
    }

    pub fn estimate_vram_usage(&self) -> (f32, String, egui::Color32) {
        // Approximate calculation
        // Params: Layers * 12 * Dim^2 (Rough Llama approximation)
        let params = (self.layers as f64) * 12.0 * (self.model_dim as f64).powi(2);

        // Memory per param: Weights (4) + Gradients (4) + Optimizer (8) = 16 bytes
        let model_mem_bytes = params * 16.0;

        // Activations (rough approximation)
        let activation_bytes = (self.batch_size as f64)
            * (self.context_len as f64)
            * (self.model_dim as f64)
            * (self.layers as f64)
            * 12.0;

        // Overhead
        let overhead_bytes = 512.0 * 1024.0 * 1024.0; // 512MB

        let total_mb = (model_mem_bytes + activation_bytes + overhead_bytes) / (1024.0 * 1024.0);

        let (status, color) = if total_mb < 8000.0 {
            ("Safe (< 8GB)", egui::Color32::GREEN)
        } else if total_mb < 16000.0 {
            ("Moderate (< 16GB)", egui::Color32::from_rgb(255, 165, 0)) // Orange
        } else if total_mb < 24000.0 {
            ("High (Requires 24GB)", egui::Color32::from_rgb(255, 69, 0)) // Red-Orange
        } else {
            ("Critical (> 24GB)", egui::Color32::RED)
        };

        (total_mb as f32, status.to_string(), color)
    }
}
