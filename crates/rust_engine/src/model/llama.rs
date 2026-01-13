//! `BitLlama` and Llama - Full model implementation

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
// use fs2::FileExt; // Implicitly used? Or compiler bug. Keeping commented to silence warning.
use std::path::Path;
use tokenizers::Tokenizer;

use crate::layers::{RMSNorm, TensorExt};
use crate::model::{BitLlamaBlock, BitLlamaConfig};

/// Epsilon for `RMSNorm`
const RMS_NORM_EPS: f64 = 1e-5;

/// Minimum temperature for sampling
const TEMP_MIN: f64 = 1e-6;

/// `BitLlama` model with embedding, layers, and LM head
pub struct BitLlama {
    pub embedding: candle_nn::Embedding,
    pub layers: Vec<BitLlamaBlock>,
    pub norm: RMSNorm,
    pub lm_head: candle_nn::Linear,
    #[allow(dead_code)]
    pub config: BitLlamaConfig,
}

impl BitLlama {
    pub fn load(cfg: &BitLlamaConfig, vb: VarBuilder<'_>) -> Result<Self> {
        // Determine primary and secondary devices
        // Ideally, `vb.device()` is the main device (likely GPU if set up that way),
        // but for hybrid, we usually start with CPU vb and move to GPU.
        // Let's assume `vb` is on CPU (Safetensors default).

        let main_device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let cpu_device = Device::Cpu;

        let n_gpu = match cfg.n_gpu_layers {
            Some(n) => n,
            None => {
                // Auto-Config Strategy
                if main_device.is_cuda() {
                    match crate::device_utils::get_vram_info(0) {
                        Ok((free, total)) => {
                            let possible = cfg.calculate_auto_offload(free);
                            println!(
                                "[Auto-Config] Detected VRAM: {} MB Free / {} MB Total",
                                free / 1024 / 1024,
                                total / 1024 / 1024
                            );
                            println!("[Auto-Config] Strategy: {} Layers on GPU / {} on CPU. (Safety Margin: 2GB)", possible, cfg.num_layers.saturating_sub(possible));
                            possible
                        }
                        Err(e) => {
                            eprintln!(
                                "[Auto-Config] Failed to detect VRAM: {e}. Defaulting to CPU."
                            );
                            0
                        }
                    }
                } else {
                    0 // CPU mode
                }
            }
        };

        let embedding = candle_nn::embedding(cfg.vocab_size, cfg.hidden_dim, vb.pp("embed"))?;
        // Embedding usually stays on GPU (or main device) as it's the entry point.
        // Or if we want full hybrid flexibility, maybe CPU?
        // Let's put embedding on main_device for now.
        // Note: candle_nn::embedding loads weights to vb's device.
        // We might need to manually move it if vb is CPU and we want GPU.
        // But `candle_nn::embedding` returns an Embedding struct which holds a Tensor.
        // We can cast `embedding.embeddings()` to device.
        // ACTUALLY: `candle_nn::embedding` just wraps the tensor lookup.

        // Let's be explicit:
        // We will respect `vb`'s device for the loading, but then move layers to their target device.
        // BUT, `BitLlamaBlock::load` now takes `device` and does `to_device`.
        // So we just need to determine the target device for each layer.

        let mut layers = Vec::new();
        for i in 0..cfg.num_layers {
            let target_device = if i < n_gpu { &main_device } else { &cpu_device };

            let layer = BitLlamaBlock::load(
                cfg.hidden_dim,
                cfg.inner_lr,
                vb.pp(format!("layers.{i}")),
                target_device,
            )?;
            layers.push(layer);
        }

        let norm = RMSNorm::load(cfg.hidden_dim, RMS_NORM_EPS, vb.pp("norm_f"), &main_device)?;
        let lm_head = candle_nn::linear_no_bias(cfg.hidden_dim, cfg.vocab_size, vb.pp("lm_head"))?;
        // lm_head weights to main_device?
        // candle_nn::linear_no_bias loads to vb device.
        // We might want to move it.
        // But `candle_nn::Linear` struct is simple.
        // Let's re-create it or move weights?
        // `candle_nn::Linear` fields are public? No, `weight` is private, accessed via `weight()`.
        // Let's just assume for now we keep it as returned by `linear_no_bias`.
        // Ideally, the final layer should be on GPU for fast logits.

        Ok(Self { embedding, layers, norm, lm_head, config: cfg.clone() })
    }

    pub fn precompute_packed(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.precompute_packed()?;
        }
        Ok(())
    }

    /// Forward for single token (inference)
    #[allow(dead_code)]
    pub fn forward_one(&self, x: &Tensor, w_states: &mut [Tensor]) -> Result<Tensor> {
        let mut h = self.embedding.forward(x)?.squeeze(0)?;

        for (i, layer) in self.layers.iter().enumerate() {
            // Hybrid Logic:
            // Input `h` might be on GPU, Layer might be on CPU.
            // Check layer device (heuristic: use norm1.weight device)
            let layer_device = layer.norm1.weight.device();

            // Move input to layer's device if needed
            let h_device = h.device();
            let h_in = if h_device.same_device(layer_device) {
                std::borrow::Cow::Borrowed(&h)
            } else {
                std::borrow::Cow::Owned(h.to_device(layer_device)?)
            };

            // W state also needs to be on correct device
            let w_state = &w_states[i];
            let w_device = w_state.device();
            let w_in = if w_device.same_device(layer_device) {
                std::borrow::Cow::Borrowed(w_state)
            } else {
                std::borrow::Cow::Owned(w_state.to_device(layer_device)?)
            };

            let (h_new, w_new) = layer.forward(&h_in, &w_in)?;

            // Update w_states[i] with new state (keeping it on layer device for next step?)
            // Usually state stays where the layer is.
            w_states[i] = w_new;

            h = h_new;
        }

        let h_norm = self.norm.forward(&h)?;
        let w = self.lm_head.weight();
        let logits = h_norm.matmul_robust(&w.t()?)?;
        Ok(logits)
    }

    /// Forward chunkwise (parallel training)
    pub fn forward_chunkwise(
        &self,
        x: &Tensor,
        w_states: &mut [Tensor],
        chunk_size: usize,
    ) -> Result<Tensor> {
        let mut h = self.embedding.forward(x)?;

        for (i, layer) in self.layers.iter().enumerate() {
            // Hybrid Logic: Ensure input 'h' is on the layer's device
            let layer_device = layer.norm1.weight.device();
            let h_device = h.device();

            let h_in = if h_device.same_device(layer_device) {
                std::borrow::Cow::Borrowed(&h)
            } else {
                std::borrow::Cow::Owned(h.to_device(layer_device)?)
            };

            // w_states[i] must also be on layer device
            let w_state = &w_states[i];
            let w_device = w_state.device();
            let w_in = if w_device.same_device(layer_device) {
                std::borrow::Cow::Borrowed(w_state)
            } else {
                std::borrow::Cow::Owned(w_state.to_device(layer_device)?)
            };

            let (h_new, w_final) = layer.forward_chunkwise(&h_in, &w_in, chunk_size)?;

            // w_final is on layer_device. Store it back.
            // If we want w_states to stay on their respective devices, this is fine.
            w_states[i] = w_final;

            // h propagates on this device. Next layer will move it if needed.
            h = h_new;
        }

        let h_norm = self.norm.forward(&h)?;
        let w = self.lm_head.weight();
        let logits = h_norm.matmul_robust(&w.t()?)?;

        Ok(logits)
    }

    pub fn collect_tensors(&self) -> std::collections::HashMap<String, Tensor> {
        let mut tensors = std::collections::HashMap::new();
        tensors.insert("embed.weight".to_string(), self.embedding.embeddings().clone());

        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{i}");
            tensors.insert(format!("{prefix}.norm1.weight"), layer.norm1.weight.clone());
            tensors
                .insert(format!("{prefix}.ttt.down.weight"), layer.ttt.proj_down.weight.clone());
            tensors.insert(format!("{prefix}.ttt.up.weight"), layer.ttt.proj_up.weight.clone());
            tensors.insert(format!("{prefix}.norm2.weight"), layer.norm2.weight.clone());
            tensors.insert(format!("{prefix}.mlp.gate_proj.weight"), layer.mlp.w1.weight.clone());
            tensors.insert(format!("{prefix}.mlp.down_proj.weight"), layer.mlp.w2.weight.clone());
            tensors.insert(format!("{prefix}.mlp.up_proj.weight"), layer.mlp.w3.weight.clone());
        }

        tensors.insert("norm_f.weight".to_string(), self.norm.weight.clone());
        tensors.insert("lm_head.weight".to_string(), self.lm_head.weight().clone());

        tensors
    }
}

/// High-level Llama API with tokenizer and state management
pub struct Llama {
    pub model: BitLlama,
    pub tokenizer: Tokenizer,
    pub device: candle_core::Device,
    pub w_states: Vec<Tensor>,
    /// Holds the shared lock on the model file to prevent modification during use
    pub _lock_file: Option<std::fs::File>,
}

impl Llama {
    /// Auto-detect format and load model
    pub fn load_auto<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let device = candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);
        Self::load_auto_with_device(path, &device)
    }

    pub fn load_auto_with_device<P: AsRef<Path>>(
        path: P,
        device: &candle_core::Device,
    ) -> anyhow::Result<Self> {
        let path = path.as_ref();

        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "bitt" {
                    return Self::from_bitt_file_with_device(path, device);
                } else if ext == "safetensors" {
                    let parent = path.parent().unwrap_or(Path::new("."));
                    return Self::new_with_weights(parent, path, device);
                }
            }
        } else if path.is_dir() {
            return Self::new_with_device(path, device);
        }

        Self::from_bitt_file_with_device(path, device)
            .or_else(|_| Self::new_with_device(path, device))
            .map_err(|_| {
                anyhow::anyhow!(
                    "Failed to load model from {path:?}. Not a valid .bitt file or legacy directory."
                )
            })
    }

    /// Load from .bitt native container
    pub fn from_bitt_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let device = candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);
        Self::from_bitt_file_with_device(path, &device)
    }

    pub fn from_bitt_file_with_device<P: AsRef<Path>>(
        path: P,
        device: &candle_core::Device,
    ) -> anyhow::Result<Self> {
        let path = path.as_ref();

        // 🔒 Phase 2: Shared Lock
        let lock_path = format!("{}.lock", path.display());
        let lock_file_handle =
            std::fs::File::open(&lock_path).or_else(|_| std::fs::File::create(&lock_path)).ok();

        if let Some(ref f) = lock_file_handle {
            if let Err(e) = f.lock_shared() {
                tracing::warn!("Failed to acquire shared lock on {}: {}", lock_path, e);
            }
        }

        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        if mmap.len() < 12 || &mmap[0..4] != b"BITT" {
            anyhow::bail!("Invalid format: Not a .bitt file (Magic mismatch)");
        }

        let header_len_bytes: [u8; 8] = mmap[4..12].try_into()?;
        let header_len = u64::from_le_bytes(header_len_bytes) as usize;

        let header_start = 12;
        let header_end = header_start + header_len;
        if mmap.len() < header_end {
            anyhow::bail!("Invalid format: File too short for header");
        }

        let header_slice = &mmap[header_start..header_end];
        let header_json: serde_json::Value = serde_json::from_slice(header_slice)?;

        let config: BitLlamaConfig = serde_json::from_value(header_json["config"].clone())
            .map_err(|e| anyhow::anyhow!("Failed to parse config from BITT header: {e}"))?;

        use std::str::FromStr;
        let tokenizer_json_str = header_json["tokenizer"].to_string();
        let tokenizer = Tokenizer::from_str(&tokenizer_json_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse tokenizer from BITT header: {e}"))?;

        let body_start = header_end;
        let body_slice = &mmap[body_start..];

        let vb = candle_nn::VarBuilder::from_buffered_safetensors(
            body_slice.to_vec(),
            DType::F32,
            device,
        )?;

        let mut model = BitLlama::load(&config, vb)?;
        model.precompute_packed()?;

        let d_small = config.hidden_dim / 4;
        let mut w_states = Vec::new();
        for _ in 0..config.num_layers {
            let w = Tensor::zeros((d_small, d_small), DType::F32, device)?;
            w_states.push(w);
        }

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            w_states,
            _lock_file: lock_file_handle,
        })
    }

    pub fn new<P: AsRef<Path>>(model_dir: P) -> anyhow::Result<Self> {
        let device = candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);
        Self::new_with_device(model_dir, &device)
    }

    pub fn new_with_device<P: AsRef<Path>>(
        model_dir: P,
        device: &candle_core::Device,
    ) -> anyhow::Result<Self> {
        let dir = model_dir.as_ref();
        Self::new_with_weights(dir, &dir.join("model.safetensors"), device)
    }

    pub fn new_with_weights<P: AsRef<Path>>(
        model_dir: P,
        weights_path: P,
        device: &candle_core::Device,
    ) -> anyhow::Result<Self> {
        let dir = model_dir.as_ref();
        let weights = weights_path.as_ref();

        // 🔒 Phase 2: Shared Lock
        let lock_path = format!("{}.lock", weights.display());
        let lock_file_handle =
            std::fs::File::open(&lock_path).or_else(|_| std::fs::File::create(&lock_path)).ok();

        if let Some(ref f) = lock_file_handle {
            if let Err(e) = f.lock_shared() {
                tracing::warn!("Failed to acquire shared lock on {}: {}", lock_path, e);
            }
        }

        let config_path = dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
            anyhow::anyhow!("Failed to read config.json from {config_path:?}: {e}")
        })?;
        let config: BitLlamaConfig = serde_json::from_str(&config_str)?;

        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            anyhow::anyhow!("Failed to load tokenizer.json from {tokenizer_path:?}: {e}")
        })?;

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)?
        };

        let mut model = BitLlama::load(&config, vb)?;
        model.precompute_packed()?;

        let d_small = config.hidden_dim / 4;
        let mut w_states = Vec::new();
        for _ in 0..config.num_layers {
            let w = Tensor::zeros((d_small, d_small), DType::F32, device)?;
            w_states.push(w);
        }

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            w_states,
            _lock_file: lock_file_handle,
        })
    }

    /// Stream completion with callback
    pub fn stream_completion<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temp: f64,
        mut callback: F,
    ) -> anyhow::Result<String>
    where
        F: FnMut(&str) -> anyhow::Result<bool>,
    {
        let temperature = if temp <= 0.0 { TEMP_MIN } else { temp };

        let encoding = self.tokenizer.encode(prompt, true).map_err(|e| anyhow::anyhow!(e))?;
        let tokens = encoding.get_ids();

        if tokens.is_empty() {
            return Ok(String::new());
        }

        let mut all_tokens = tokens.to_vec();
        let mut next_token = *all_tokens.last().unwrap();

        for &t in tokens {
            let input = Tensor::new(&[t], &self.device)?;
            let logits = self.model.forward_one(&input, &mut self.w_states)?;

            if t == *tokens.last().unwrap() {
                let logits_v = logits.squeeze(0)?;
                let prs = candle_nn::ops::softmax(&(&logits_v / temperature)?, 0)?;
                let prs_vec = prs.to_vec1::<f32>()?;
                next_token = Self::sample_multinomial(&prs_vec)?;
            }
        }

        all_tokens.push(next_token);

        let mut generated_tokens: Vec<u32> = vec![next_token];
        let mut prev_decoded =
            self.tokenizer.decode(&generated_tokens, true).map_err(|e| anyhow::anyhow!(e))?;

        if !prev_decoded.is_empty() && !callback(&prev_decoded)? {
            let full_text =
                self.tokenizer.decode(&all_tokens, true).map_err(|e| anyhow::anyhow!(e))?;
            return Ok(full_text);
        }

        for _ in 0..(max_tokens - 1) {
            let input = Tensor::new(&[next_token], &self.device)?;
            let logits = self.model.forward_one(&input, &mut self.w_states)?;
            let logits_v = logits.squeeze(0)?;

            let prs = candle_nn::ops::softmax(&(&logits_v / temperature)?, 0)?;
            let prs_vec = prs.to_vec1::<f32>()?;
            next_token = Self::sample_multinomial(&prs_vec)?;

            all_tokens.push(next_token);
            generated_tokens.push(next_token);

            let current_decoded =
                self.tokenizer.decode(&generated_tokens, true).map_err(|e| anyhow::anyhow!(e))?;

            let delta = if current_decoded.len() > prev_decoded.len() {
                &current_decoded[prev_decoded.len()..]
            } else {
                &current_decoded
            };

            let processed_delta = Self::process_escape_sequences(delta);

            if !processed_delta.is_empty() && !callback(&processed_delta)? {
                break;
            }
            prev_decoded = current_decoded;
        }

        let full_text = self.tokenizer.decode(&all_tokens, true).map_err(|e| anyhow::anyhow!(e))?;
        Ok(full_text)
    }

    fn process_escape_sequences(delta: &str) -> String {
        if !delta.contains("\\x") {
            return delta.to_string();
        }

        let mut result = String::new();
        let mut chars = delta.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\\' {
                if chars.peek() == Some(&'x') {
                    chars.next();
                    let hex: String = chars.by_ref().take(2).collect();
                    if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                        result.push(byte as char);
                    }
                } else {
                    result.push(c);
                }
            } else {
                result.push(c);
            }
        }
        result
    }

    pub fn export_to_bitt<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let path = path.as_ref();
        use std::io::Write;

        let tensors = self.model.collect_tensors();
        let temp_path = path.with_extension("temp.safetensors");
        candle_core::safetensors::save(&tensors, &temp_path)?;
        let safetensors_bytes = std::fs::read(&temp_path)?;
        let _ = std::fs::remove_file(temp_path);

        let config_json = serde_json::to_value(self.model.config)?;
        let tokenizer_json_str = self.tokenizer.to_string(false).map_err(|e| anyhow::anyhow!(e))?;

        let header = serde_json::json!({
            "config": config_json,
            "tokenizer": tokenizer_json_str
        });
        let header_vec = serde_json::to_vec(&header)?;
        let header_len = header_vec.len() as u64;

        let mut file = std::fs::File::create(path)?;
        file.write_all(b"BITT")?;
        file.write_all(&header_len.to_le_bytes())?;
        file.write_all(&header_vec)?;
        file.write_all(&safetensors_bytes)?;

        Ok(())
    }

    pub fn reset_state(&mut self) -> anyhow::Result<()> {
        let d_small = self.model.config.hidden_dim / 4;
        for w in &mut self.w_states {
            *w = Tensor::zeros((d_small, d_small), DType::F32, &self.device)?;
        }
        Ok(())
    }

    fn sample_multinomial(probs: &[f32]) -> anyhow::Result<u32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let r = rng.gen::<f32>();
        let mut cdf = 0.0;
        for (i, p) in probs.iter().enumerate() {
            cdf += p;
            if r < cdf {
                return Ok(i as u32);
            }
        }
        Ok((probs.len() - 1) as u32)
    }
}
