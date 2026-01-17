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
    pub n_heads: usize,
    #[serde(default)]
    pub n_kv_heads: Option<usize>, // Added for GQA
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
    pub profile: String, // "consumer" (8GB) or "server" (24GB+)

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

    // RoPE / Positional Embeddings
    #[serde(default = "default_rope")]
    pub rope_theta: f64,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,

    // Phase 12: MeZO & Instruct
    #[serde(default)]
    pub use_mezo: bool,
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,
    #[serde(default)]
    pub instruct_path: String,
    #[serde(default, skip)]
    pub mock: bool,
    #[serde(default)]
    pub lm_head_cpu: bool,
}

fn default_input_pattern() -> String {
    "data/*".to_string()
}
fn default_temp() -> f64 {
    0.8
}
fn default_max_tokens() -> usize {
    100
}
fn default_accum_steps() -> usize {
    1
}
fn default_profile() -> String {
    "consumer".to_string()
}
fn default_epsilon() -> f64 {
    1e-3
}
fn default_rope() -> f64 {
    10000.0
}
fn default_max_pos() -> usize {
    2048
}

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
            n_kv_heads: None, // Default to MHA (Same as n_heads)
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
            use_mezo: false,
            epsilon: 1e-3,
            instruct_path: "".to_string(),
            mock: false,
            rope_theta: default_rope(),
            max_position_embeddings: default_max_pos(),
            lm_head_cpu: false, // Default to GPU
        }
    }
}

impl ProjectConfig {
    pub fn from_args(args: &crate::train::TrainArgs) -> Self {
        Self {
            name: "Training Run".to_string(),
            created_at: chrono::Local::now().to_string(),
            vocab_size: 8000, // Should be updated manually if different or inferred
            model_type: ModelType::Unigram,
            val_ratio: 0.05,
            model_dim: args.dim,
            layers: args.layers,
            context_len: args.context_len,
            n_heads: 8,
            n_kv_heads: None, // Default to MHA
            batch_size: args.batch_size,
            steps: args.steps,
            lr: args.lr,
            min_lr: args.min_lr,
            warmup_steps: args.warmup_steps,
            save_interval: args.save_interval,
            accum_steps: args.accum.max(1),
            profile: "consumer".to_string(),
            input_pattern: "N/A".to_string(),
            template: "".to_string(),
            use_template: false,
            inference_temp: default_temp(),
            inference_max_tokens: default_max_tokens(),
            use_mezo: false, // Default context
            epsilon: args.epsilon,
            instruct_path: "".to_string(),
            mock: args.mock,
            rope_theta: default_rope(),
            max_position_embeddings: args.context_len.max(2048),
            lm_head_cpu: false,
        }
    }

    pub fn to_bit_llama_config(&self, inner_lr: f64) -> cortex_rust::BitLlamaConfig {
        cortex_rust::BitLlamaConfig {
            arch: cortex_rust::ModelArch::TTT, // Default to TTT for trainer for now
            vocab_size: self.vocab_size,
            hidden_dim: self.model_dim,
            num_layers: self.layers,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads.unwrap_or(self.n_heads),
            intermediate_dim: None,
            inner_lr,
            n_gpu_layers: None,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            lm_head_cpu: self.lm_head_cpu,
        }
    }

    pub fn estimate_vram_usage(&self) -> (f32, String, egui::Color32) {
        let eff = self.estimate_efficiency();
        (eff.bit_ttt_mb as f32, eff.status, eff.color)
    }

    pub fn estimate_efficiency(&self) -> VramEfficiencyMetrics {
        // Param Count (Approximation for Llama)
        let d = self.model_dim as f64;
        let l = self.layers as f64;
        let v = self.vocab_size as f64;

        // Params
        let params_layer = 16.0 * d * d;
        let params_embed = 2.0 * v * d; // Token Embed + Output Head
        let total_params = (l * params_layer) + params_embed;

        // --- Bit-TTT Cost (Inference / Deployment) ---
        // VRAM = Weights(1.58bit) + KV Cache + Overhead
        // 1.58 bits ≈ 0.25 bytes (packed).
        // We add some buffer for quantization metadata/scales (~0.05).
        let bytes_bitttt = 0.3;

        // --- FP16 Transformer Cost (Inference / Deployment) ---
        // VRAM = Weights(16bit) + KV Cache + Overhead
        // 16 bits = 2.0 bytes.
        let bytes_fp16 = 2.0;

        // Context / Activation (KV Cache for Inference)
        // KV Size = 2(K+V) * Layers * (KV_Dim) * Context * (Bytes per Element)
        // FP16 KV = 2 bytes.
        let ctx_len = self.context_len as f64;
        let d = self.model_dim as f64;
        let l = self.layers as f64;
        let heads = self.n_heads.max(1) as f64;
        let kv_heads = self.n_kv_heads.unwrap_or(self.n_heads).max(1) as f64;

        // Effective KV dimension due to GQA/MQA
        // If n_kv_heads < n_heads, the key/value states are shared across heads
        let kv_dim = (d / heads) * kv_heads;

        let kv_bytes_per_token = 2.0 * l * kv_dim * 2.0; // 2(K+V) * Layers * KV_Dim * 2(FP16)
        let kv_total_mb = (kv_bytes_per_token * ctx_len) / (1024.0 * 1024.0);

        // Totals
        let model_mb_bitttt = (total_params * bytes_bitttt) / (1024.0 * 1024.0);
        let model_mb_fp16 = (total_params * bytes_fp16) / (1024.0 * 1024.0);

        let overhead_mb = 256.0; // Runtime overhead (CUDA/PyTorch/Candle)

        let total_bitttt = if self.use_mezo {
            // MeZO Training (O(1) Memory): Model + Small Buffer (No KV Cache/Graph)
            model_mb_bitttt + overhead_mb + 128.0
        } else {
            // Standard Inference / Validation Cost
            model_mb_bitttt + kv_total_mb + overhead_mb
        };

        let total_fp16 = model_mb_fp16 + kv_total_mb + overhead_mb;

        let (status, color) = if total_bitttt < 8000.0 {
            ("Safe (< 8GB)", egui::Color32::GREEN)
        } else if total_bitttt < 12000.0 {
            ("Moderate (< 12GB)", egui::Color32::from_rgb(255, 165, 0))
        } else if total_bitttt < 24000.0 {
            ("High (< 24GB)", egui::Color32::from_rgb(255, 69, 0))
        } else {
            ("Critical (> 24GB)", egui::Color32::RED)
        };

        VramEfficiencyMetrics {
            bit_ttt_mb: total_bitttt,
            fp16_mb: total_fp16,
            saved_mb: total_fp16 - total_bitttt,
            saved_ratio: if total_bitttt > 0.0 {
                total_fp16 / total_bitttt
            } else {
                0.0
            },
            status: status.to_string(),
            color,
        }
    }
}

pub struct VramEfficiencyMetrics {
    pub bit_ttt_mb: f64,
    pub fp16_mb: f64,
    pub saved_mb: f64,
    pub saved_ratio: f64,
    pub status: String,
    pub color: egui::Color32,
}
