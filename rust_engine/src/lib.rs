//! Cortex Rust Engine
//!
//! Core implementation of the Bit-Llama model with TTT (Test-Time Training) support.
//! Provides both native Rust bindings and optional Python bindings via PyO3.

#![allow(non_local_definitions)]

pub mod core_engine;
pub mod legacy;

// Legacy re-exports (Deprecated: use core_engine types instead)
#[deprecated(since = "0.2.0", note = "Use cortex_rust::CandleTTTLayer instead")]
pub use legacy::bit_linear::BitLinear;
#[deprecated(since = "0.2.0", note = "Use cortex_rust::CandleTTTLayer instead")]
pub use legacy::ttt_layer::TTTLayer;

// New Core Engine re-exports
pub use core_engine::{BitLlama, BitLlamaBlock, BitLlamaConfig, Llama, TTTLayer as CandleTTTLayer};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn cortex_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BitLlamaConfig>()?;
    m.add_class::<core_engine::PyBitLlama>()?;
    Ok(())
}
