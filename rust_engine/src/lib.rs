#![allow(non_local_definitions)]

pub mod bit_linear;
pub mod c_api;
pub mod core_engine;
pub mod ttt_layer;

// Legacy re-exports (Deprecated: use core_engine types instead)
#[deprecated(since = "0.2.0", note = "Use cortex_rust::CandleTTTLayer instead")]
pub use bit_linear::BitLinear;
#[deprecated(since = "0.2.0", note = "Use cortex_rust::CandleTTTLayer instead")]
pub use ttt_layer::TTTLayer;

// New Core Engine re-exports
pub use core_engine::{BitLlama, BitLlamaBlock, BitLlamaConfig, TTTLayer as CandleTTTLayer};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn cortex_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BitLlamaConfig>()?;
    m.add_class::<core_engine::PyBitLlama>()?;
    Ok(())
}
