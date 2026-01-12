//! BitLlamaConfig - Model configuration

use serde::Deserialize;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Model configuration for BitLlama
#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone, Copy, Debug, Deserialize, serde::Serialize)]
pub struct BitLlamaConfig {
    #[pyo3(get, set)]
    pub vocab_size: usize,
    #[pyo3(get, set)]
    pub hidden_dim: usize,
    #[pyo3(get, set)]
    #[serde(alias = "n_layers")]
    pub num_layers: usize,
    #[pyo3(get, set)]
    pub inner_lr: f64,
    #[pyo3(get, set)]
    pub n_gpu_layers: Option<usize>,
}

#[cfg(feature = "python")]
#[pymethods]
impl BitLlamaConfig {
    #[new]
    pub fn new(vocab_size: usize, hidden_dim: usize, num_layers: usize, inner_lr: f64) -> Self {
        Self {
            vocab_size,
            hidden_dim,
            num_layers,
            inner_lr,
            n_gpu_layers: None,
        }
    }
}

/// Model configuration for BitLlama (Rust-only version)
#[cfg(not(feature = "python"))]
#[derive(Clone, Copy, Debug, Deserialize, serde::Serialize)]
pub struct BitLlamaConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    #[serde(alias = "n_layers")]
    pub num_layers: usize,
    pub inner_lr: f64,
    pub n_gpu_layers: Option<usize>,
}

#[cfg(not(feature = "python"))]
impl BitLlamaConfig {
    pub fn new(vocab_size: usize, hidden_dim: usize, num_layers: usize, inner_lr: f64) -> Self {
        Self {
            vocab_size,
            hidden_dim,
            num_layers,
            inner_lr,
            n_gpu_layers: None,
        }
    }
}

impl BitLlamaConfig {
    /// Calculate how many layers can fit in GPU VRAM
    /// Formula: (FreeVRAM * 0.9 - 2GB) / (Weight + KV)
    pub fn calculate_auto_offload(&self, free_vram: usize) -> usize {
        let f_free = free_vram as f64 * 0.9;
        let safety = 2.0 * 1024.0 * 1024.0 * 1024.0; // 2GB Safety Margin
        if f_free <= safety {
            return 0;
        }
        let available = (f_free - safety) as usize;

        let k = self.hidden_dim;
        // Weight: 7 matrices (Q,K,V,O, Gate,Up,Down). KxK. 2-bit (0.25 bytes).
        // Size = 7 * K^2 / 4
        let weight_bytes = (7 * k * k) / 4;

        // KV Cache: 2 (K+V) * Seq(4096) * Hidden * 2 (f16)
        // Worst case MHA (KV_Heads * Head_Dim = Hidden).
        let seq_len = 4096;
        let kv_bytes = 2 * k * seq_len * 2;

        // RMSNorm (2 * K * 4) + minimal overhead
        let overhead = k * 8 + 1024 * 1024; // 1MB misc

        let layer_size = weight_bytes + kv_bytes + overhead;



        // Log basic info (via tracing or println if tracing not init)
        // But this is pure func if possible.
        // We return count.

        let count = available / layer_size;
        if count >= self.num_layers {
            self.num_layers
        } else {
            count
        }
    }
}

