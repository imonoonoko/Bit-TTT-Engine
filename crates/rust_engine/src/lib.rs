//! Cortex Rust Engine
//!
//! Core implementation of the Bit-Llama model with TTT (Test-Time Training) support.
//! Provides both native Rust bindings and optional Python bindings via PyO3.

#![allow(non_local_definitions)]

// Core modules (Rust 2018+ style)
pub mod layers;
pub mod model;
pub mod python;

// Legacy module (deprecated)
pub mod legacy;

// Legacy re-exports (Deprecated: use model types instead)
#[deprecated(since = "0.2.0", note = "Use cortex_rust::model::layers instead")]
pub use legacy::bit_linear::BitLinear as LegacyBitLinear;
#[deprecated(since = "0.2.0", note = "Use cortex_rust::model::layers instead")]
pub use legacy::ttt_layer::TTTLayer as LegacyTTTLayer;

// Primary public API re-exports
pub use layers::{BitLinear, RMSNorm, SwiGLU, TTTLayer};
pub use model::{BitLlama, BitLlamaBlock, BitLlamaConfig, Llama};

// Alias for backward compatibility
pub use model::TTTLayer as CandleTTTLayer;

// Python module registration
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn cortex_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<model::BitLlamaConfig>()?;
    m.add_class::<python::PyBitLlama>()?;
    Ok(())
}
