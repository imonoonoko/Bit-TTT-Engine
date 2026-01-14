//! Python Bindings for BitLlama (PyO3)

#[cfg(feature = "python")]
use candle_core::{DType, Tensor, Var};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use crate::model::{BitLlama, BitLlamaConfig};
#[cfg(feature = "python")]
use crate::optim::schedule_free::{ParamsScheduleFree, ScheduleFreeOptimizer};
#[cfg(feature = "python")]
use candle_nn::VarMap;

/// Python wrapper for BitLlama model (Inference)
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
            .precompute_packed()
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

    #[pyo3(signature = (prompt, max_tokens))]
    pub fn generate(&mut self, py: Python, prompt: &str, max_tokens: usize) -> PyResult<String> {
        let _ = (prompt, max_tokens);
        py.allow_threads(move || {
            Ok("Not implemented: need tokenizer access. Use generate_tokens".to_string())
        })
    }

    pub fn generate_tokens(
        &mut self,
        py: Python,
        start_tokens: Vec<u32>,
        max_new_tokens: usize,
    ) -> PyResult<Vec<u32>> {
        py.allow_threads(move || {
            let device = self.inner.embedding.embeddings().device();
            let mut current_tokens = start_tokens.clone();

            for _ in 0..max_new_tokens {
                let last_token = *current_tokens
                    .last()
                    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Empty start tokens"))?;

                let input = Tensor::new(&[last_token], device)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                let logits = self
                    .inner
                    .forward_one(&input, &mut self.w_states)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                let logits_v = logits
                    .squeeze(0)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                // Argmax for simplicity in this MVP
                let next_token = logits_v
                    .argmax(0)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    .to_scalar::<u32>()
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                current_tokens.push(next_token);
            }

            Ok(current_tokens)
        })
    }
}

/// Python wrapper for BitLlama model (Training)
#[cfg(feature = "python")]
#[pyclass(name = "PyTrainer")]
pub struct PyTrainer {
    model: BitLlama,
    varmap: VarMap,
    optimizer: ScheduleFreeOptimizer,
    sorted_vars: Vec<Var>, // For deterministic gradient ordering
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTrainer {
    #[new]
    #[pyo3(signature = (config, checkpoint_path=None, device=None))]
    pub fn new(
        config: BitLlamaConfig,
        checkpoint_path: Option<&str>,
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

        let mut varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        // Note: We use clone() heavily for config here.
        let model = BitLlama::load(config, vb)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Load Weights if provided
        if let Some(path) = checkpoint_path {
            varmap.load(path).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load params: {}", e))
            })?;
        }

        // Initialize Optimizer
        // CRITICAL: Sort variables by name to ensure stable order for optimizer mapping
        let data = varmap.data().lock().unwrap();
        let mut named_vars: Vec<_> = data.iter().map(|(n, v)| (n.clone(), v.clone())).collect();
        // Drop lock before sorting/processing to minimize hold time
        drop(data);

        named_vars.sort_by(|a, b| a.0.cmp(&b.0));

        let vars: Vec<Var> = named_vars.iter().map(|(_, v)| v.clone()).collect();
        let sorted_vars = vars.clone();

        let params = ParamsScheduleFree {
            lr: 0.002,
            ..Default::default()
        };
        let optimizer = ScheduleFreeOptimizer::new(vars, params)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            model,
            varmap,
            optimizer,
            sorted_vars,
        })
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.optimizer.set_learning_rate(lr);
    }

    #[pyo3(signature = (py_input_ids, py_targets))]
    pub fn train_step(
        &mut self,
        py: Python,
        py_input_ids: Vec<u32>,
        py_targets: Vec<u32>,
    ) -> PyResult<f64> {
        py.allow_threads(move || {
            let device = self.model.embedding.embeddings().device();
            let input_tensor = Tensor::new(py_input_ids.as_slice(), device)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                .unsqueeze(0) // Batch dim 1
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let target_tensor = Tensor::new(py_targets.as_slice(), device)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            // 1. Pre-step
            self.optimizer
                .pre_step()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            // 2. Forward
            // Create ephemeral w_states (zeroed) for this chunk
            let d_small = self.model.config.hidden_dim / 4;
            let mut w_states = Vec::new();
            for _ in 0..self.model.config.num_layers {
                let w = Tensor::zeros((d_small, d_small), DType::F32, device)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                w_states.push(w);
            }

            let seq_len = py_input_ids.len();

            let logits = self
                .model
                .forward_chunkwise(&input_tensor, &mut w_states, seq_len)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            // 3. Loss
            let b_sz = 1;
            let logits = logits
                .reshape((b_sz * seq_len, logits.dim(2).unwrap()))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let targets = target_tensor
                .reshape((b_sz * seq_len,))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            let loss = candle_nn::loss::cross_entropy(&logits, &targets)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            // 4. Backward
            let grads_store = loss
                .backward()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            // 5. Collect Gradients in determinstic order
            let mut grad_tensors = Vec::new();
            for var in &self.sorted_vars {
                if let Some(g) = grads_store.get(var) {
                    grad_tensors.push(g.clone());
                } else {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "Missing gradient for a variable. Graph disconnected?",
                    ));
                }
            }

            // 6. Optimizer Step
            self.optimizer
                .step(&grad_tensors)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            // 7. Return Loss
            let loss_val = loss
                .to_scalar::<f32>()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                as f64;
            Ok(loss_val)
        })
    }

    #[pyo3(signature = (path))]
    pub fn save_checkpoint(&self, path: &str) -> PyResult<()> {
        self.varmap
            .save(path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Save Optimizer State (Z)
        // Use sorted_vars to ensure same order as self.optimizer.z
        let mut z_map = std::collections::HashMap::new();

        let data = self.varmap.data().lock().unwrap();
        // We need to map var -> name to key the map.
        // Wait, saving requires "name" -> "tensor".
        // self.sorted_vars is just tensors.
        // But `data` (HashMap) has names.
        // Efficient way:
        // Iterate `data` and find index in `sorted_vars`? No, slow.
        // Better:
        // Re-construct the sorted list of (name, var) pairs similarly to `new`.
        // Since `Var` is RefCell/Arc id based, if we sort by name again, we get same order.
        let mut named_vars: Vec<_> = data.iter().collect();
        named_vars.sort_by(|a, b| a.0.cmp(b.0));

        for (i, (name, _var)) in named_vars.iter().enumerate() {
            if i < self.optimizer.z.len() {
                z_map.insert(format!("{}.z", name), self.optimizer.z[i].clone());
            }
        }

        let optim_path = format!("{}.optim", path);
        candle_core::safetensors::save(&z_map, &optim_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(())
    }
}
