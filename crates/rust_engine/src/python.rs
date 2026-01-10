//! Python Bindings for BitLlama (PyO3)

#[cfg(feature = "python")]
use candle_core::{DType, Tensor};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use crate::model::{BitLlama, BitLlamaConfig};

/// Python wrapper for BitLlama model
#[cfg(feature = "python")]
#[pyclass(name = "BitLlama")]
pub struct PyBitLlama {
    inner: BitLlama,
    w_states: Vec<Tensor>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBitLlama {
    #[new]
    #[pyo3(signature = (config, checkpoint_path, device=None))]
    pub fn new(
        config: BitLlamaConfig,
        checkpoint_path: &str,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let device = match device {
            Some("cuda") => candle_core::Device::new_cuda(0).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("CUDA error: {}", e))
            })?,
            Some("cpu") | None => candle_core::Device::Cpu,
            Some(unknown) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unsupported device: {}. Use 'cpu' or 'cuda'",
                    unknown
                )))
            }
        };

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[checkpoint_path], DType::F32, &device)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        };

        let mut model = BitLlama::load(config, vb)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        model
            .precompute_for_inference()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let d_small = config.hidden_dim / 4;
        let mut w_states = Vec::new();
        for _ in 0..config.num_layers {
            let w = Tensor::zeros((d_small, d_small), DType::F32, &device)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            w_states.push(w);
        }

        Ok(Self {
            inner: model,
            w_states,
        })
    }

    pub fn forward(&mut self, token_id: u32) -> PyResult<Vec<f32>> {
        let device = self.inner.embedding.embeddings().device();
        let input = Tensor::new(&[token_id], device)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let logits = self
            .inner
            .forward_one(&input, &mut self.w_states)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let logits_vec = logits
            .squeeze(0)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(logits_vec)
    }
}
