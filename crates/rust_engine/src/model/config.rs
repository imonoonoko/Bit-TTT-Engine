//! BitLlamaConfig - Model configuration

use serde::Deserialize;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Model configuration for BitLlama
#[derive(Clone, Copy, Debug, Deserialize, serde::Serialize)]
#[cfg_attr(feature = "python", pyclass)]
#[derive(Default)]
pub enum ModelArch {
    #[serde(rename = "ttt")]
    #[default]
    TTT,
    #[serde(rename = "llama")]
    Llama,
}

#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone, Copy, Debug, Deserialize, serde::Serialize)]
pub struct BitLlamaConfig {
    #[pyo3(get, set)]
    #[serde(default)]
    pub arch: ModelArch,
    #[pyo3(get, set)]
    pub vocab_size: usize,
    #[pyo3(get, set)]
    #[serde(alias = "hidden_size")]
    pub hidden_dim: usize,
    #[pyo3(get, set)]
    #[serde(alias = "num_hidden_layers")]
    #[serde(alias = "n_layers")]
    pub num_layers: usize,
    #[pyo3(get, set)]
    #[serde(alias = "num_attention_heads")] // Add fields even if not exposed to Py
    pub n_heads: usize,
    #[pyo3(get, set)]
    #[serde(alias = "num_key_value_heads")]
    pub n_kv_heads: usize,
    #[pyo3(get, set)]
    #[serde(alias = "intermediate_size")]
    pub intermediate_dim: Option<usize>, // Optional, defaults to hidden*4 if None? Or explicit.
    #[pyo3(get, set)]
    #[serde(default)]
    pub inner_lr: f64,
    #[pyo3(get, set)]
    pub n_gpu_layers: Option<usize>,
    #[pyo3(get, set)]
    #[serde(default = "default_rope")]
    pub rope_theta: f64,
    #[pyo3(get, set)]
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[pyo3(get, set)]
    #[serde(default)]
    pub lm_head_cpu: bool,
}

fn default_rope() -> f64 {
    10000.0
}
fn default_max_pos() -> usize {
    2048
}

#[cfg(feature = "python")]
#[pymethods]
impl BitLlamaConfig {
    #[new]
    pub fn new(
        vocab_size: usize,
        hidden_dim: usize,
        num_layers: usize,
        inner_lr: f64,
        lm_head_cpu: Option<bool>,
    ) -> Self {
        Self {
            arch: ModelArch::TTT,
            vocab_size,
            hidden_dim,
            num_layers,
            n_heads: hidden_dim / 64,
            n_kv_heads: hidden_dim / 64,
            intermediate_dim: Some(hidden_dim * 4),
            inner_lr,
            n_gpu_layers: None,
            rope_theta: 10000.0,
            max_position_embeddings: 2048,
            lm_head_cpu: lm_head_cpu.unwrap_or(false),
        }
    }

    /// Calculate possible offload layers for given VRAM (bytes)
    /// Returns (n_gpu_layers, used_vram_mb)
    pub fn calculate_auto_offload(&self, vram_bytes: usize) -> (usize, f32) {
        // Estimate size per layer
        // Llama-3-8B: ~4GB total
        // Layers: 32
        // Size/Layer ~ 120MB (Quantized)
        // KV Cache (4096 ctx) ~ 300MB
        // Base Overhead (Embed + Head) ~ 200MB (if on GPU)

        let mb = 1024.0 * 1024.0;
        let available_mb = vram_bytes as f64 / mb;

        // Conservative constants
        let base_overhead = 500.0; // Reserve for CUDA context
        let layer_size = 130.0; // BitLinear weights + activation overhead
        let kv_cache_size = 10.0; // Per layer for standard context? Need refining.

        if available_mb < base_overhead {
            return (0, 0.0);
        }

        let usable = available_mb - base_overhead;
        let per_layer = layer_size + kv_cache_size;

        let n = (usable / per_layer).floor() as usize;
        let n = n.min(self.num_layers);

        let estimated = base_overhead + (n as f64 * per_layer);

        (n, estimated as f32)
    }
}
