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
}

#[cfg(not(feature = "python"))]
impl BitLlamaConfig {
    pub fn new(vocab_size: usize, hidden_dim: usize, num_layers: usize, inner_lr: f64) -> Self {
        Self {
            vocab_size,
            hidden_dim,
            num_layers,
            inner_lr,
        }
    }
}
