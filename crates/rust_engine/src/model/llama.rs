//! BitLlama and Llama - Full model implementation

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
// use fs2::FileExt; // Implicitly used? Or compiler bug. Keeping commented to silence warning.
use std::path::Path;
use tokenizers::Tokenizer;

use crate::layers::RMSNorm;
use crate::model::{BitLlamaBlock, BitLlamaConfig};

/// Epsilon for RMSNorm
const RMS_NORM_EPS: f64 = 1e-5;

/// Minimum temperature for sampling
const TEMP_MIN: f64 = 1e-6;

/// BitLlama model with embedding, layers, and LM head
pub struct BitLlama {
    pub embedding: candle_nn::Embedding,
    pub layers: Vec<BitLlamaBlock>,
    pub norm: RMSNorm,
    pub lm_head: candle_nn::Linear,
    pub kv_caches: Vec<Option<crate::layers::KVCache>>,
    pub current_pos: usize,
    #[allow(dead_code)]
    pub config: BitLlamaConfig,
    /// GPU device used for layers 0..n_gpu_layers (None if CPU-only mode)
    pub gpu_device: Option<Device>,
    /// CPU device for layers n_gpu_layers..num_layers and lm_head (if lm_head_cpu=true)
    pub cpu_device: Device,
    /// Number of layers on GPU (from config.n_gpu_layers)
    pub n_gpu: usize,
}

impl BitLlama {
    pub fn load(cfg: BitLlamaConfig, vb: VarBuilder) -> Result<Self> {
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
                            let (n, est_vram) = cfg.calculate_auto_offload(free);
                            println!(
                                "[Auto-Config] Detected VRAM: {} MB Free / {} MB Total",
                                free / 1024 / 1024,
                                total / 1024 / 1024
                            );
                            println!("[Auto-Config] Strategy: {} Layers on GPU / {} on CPU. (Est: {:.2} MB)", n, cfg.num_layers.saturating_sub(n), est_vram);
                            n
                        }
                        Err(e) => {
                            eprintln!(
                                "[Auto-Config] Failed to detect VRAM: {}. Defaulting to CPU.",
                                e
                            );
                            0
                        }
                    }
                } else {
                    0 // CPU mode
                }
            }
        };

        // IO layers (embedding, norm, lm_head) should be on GPU only if n_gpu > 0
        let io_device = if n_gpu > 0 { &main_device } else { &cpu_device };
        let lm_head_device = if cfg.lm_head_cpu {
            &cpu_device
        } else {
            io_device
        };

        // Support both "model.embed_tokens" (HF) and "embed" (BitLlama Legacy)
        let embedding_raw =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_dim, vb.pp("model.embed_tokens"))
                .or_else(|_| {
                    candle_nn::embedding(cfg.vocab_size, cfg.hidden_dim, vb.pp("embed"))
                })?;

        // [Plan B] Explicit Mmap Detachment for Embedding
        // If on CPU, we must Deep Copy. If on GPU, to_device copies automatically.
        let embedding = if io_device.is_cpu() {
            // Flatten 2D embedding [vocab, hidden] to 1D, then reshape
            let data = embedding_raw.embeddings().flatten_all()?.to_vec1::<f32>()?;
            let w = Tensor::from_vec(data, (cfg.vocab_size, cfg.hidden_dim), io_device)?;
            candle_nn::Embedding::new(w, cfg.hidden_dim)
        } else {
            candle_nn::Embedding::new(
                embedding_raw.embeddings().to_device(io_device)?,
                cfg.hidden_dim,
            )
        };

        let mut layers = Vec::new();
        for i in 0..cfg.num_layers {
            let target_device = if i < n_gpu { &main_device } else { &cpu_device };

            // Support "model.layers.i" (HF) and "layers.i" (Legacy)
            let layer_vb = if vb
                .contains_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i))
                || vb.contains_tensor(&format!(
                    "model.layers.{}.post_attention_layernorm.weight",
                    i
                )) {
                vb.pp(format!("model.layers.{}", i))
            } else {
                vb.pp(format!("layers.{}", i))
            };

            let layer = BitLlamaBlock::load(&cfg, layer_vb, target_device)?;
            layers.push(layer);
        }

        // Support "model.norm" (HF) and "norm_f" (Legacy)
        let norm = RMSNorm::load(cfg.hidden_dim, RMS_NORM_EPS, vb.pp("model.norm"), io_device)
            .or_else(|_| RMSNorm::load(cfg.hidden_dim, RMS_NORM_EPS, vb.pp("norm_f"), io_device))?;

        // Load LM Head and move to lm_head_device
        let lm_head_raw =
            candle_nn::linear_no_bias(cfg.hidden_dim, cfg.vocab_size, vb.pp("lm_head"))?;

        // [Hybrid Guard] Move LM Head with Deep Copy if CPU
        let lm_head = if lm_head_device.is_cpu() {
            // Fix: Flatten 2D tensor to 1D before converting to vector
            let data = lm_head_raw.weight().flatten_all()?.to_vec1::<f32>()?;
            let w = Tensor::from_vec(data, (cfg.vocab_size, cfg.hidden_dim), lm_head_device)?;
            candle_nn::Linear::new(w, None)
        } else {
            candle_nn::Linear::new(lm_head_raw.weight().to_device(lm_head_device)?, None)
        };

        Ok(Self {
            embedding,
            layers,
            norm,
            lm_head,
            kv_caches: vec![
                Some(crate::layers::KVCache::new(cfg.max_position_embeddings));
                cfg.num_layers
            ],
            current_pos: 0,
            config: cfg,
            gpu_device: if n_gpu > 0 { Some(main_device) } else { None },
            cpu_device,
            n_gpu,
        })
    }

    /// Helper to get zero states for TTT
    pub fn new_w_states(&self) -> Vec<Tensor> {
        // TTT State size: [Hidden, Hidden]
        // If Attention, we don't need w_states (they are unused), but keep API consistent.
        let device = self.embedding.embeddings().device();
        let dim = self.config.hidden_dim;
        // Optimization: Don't allocate if Attention?
        // But forward_one signature requires w_states slice.
        // Allocate zeros.
        vec![Tensor::zeros((dim, dim), DType::F32, device).unwrap(); self.layers.len()]
    }

    pub fn precompute_packed(&mut self) -> Result<()> {
        for layer in self.layers.iter_mut() {
            layer.precompute_packed()?;
        }
        Ok(())
    }

    pub fn reset_kv_cache(&mut self) {
        self.kv_caches = vec![
            Some(crate::layers::KVCache::new(
                self.config.max_position_embeddings
            ));
            self.layers.len()
        ];
        self.current_pos = 0;
    }

    /// Forward for single token (inference)
    #[allow(dead_code)]
    /// Main forward pass (dispatches to chunkwise or one)
    pub fn forward(&mut self, x: &Tensor, w_states: &mut [Tensor]) -> Result<Tensor> {
        let (_b, seq_len) = x.dims2()?;
        if seq_len > 1 {
            self.forward_chunkwise(x, w_states, seq_len)
        } else {
            self.forward_one(x, w_states)
        }
    }

    pub fn forward_one(&mut self, x: &Tensor, w_states: &mut [Tensor]) -> Result<Tensor> {
        // Ensure input is [Batch, Seq] -> [1, 1] if single token
        let x = if x.rank() == 1 {
            x.unsqueeze(0)?
        } else {
            x.clone()
        };
        let mut h = self.embedding.forward(&x)?;

        for (i, layer) in self.layers.iter().enumerate() {
            // [Hybrid Fix] Select device based on layer index and stored devices
            // Layers 0..n_gpu should be on GPU, layers n_gpu.. should be on CPU
            let target_device: &Device = if i < self.n_gpu {
                self.gpu_device.as_ref().unwrap_or(&self.cpu_device)
            } else {
                &self.cpu_device
            };

            // Move hidden state to target device
            let h_layer = if h.device().same_device(target_device) {
                h.clone()
            } else {
                // Log device transition only at boundary (Layer n_gpu)
                if i == self.n_gpu {
                    /*
                    eprintln!(
                        "🚀 [Layer {}] Moving tensor {:?} -> {:?}",
                        i,
                        h.device(),
                        target_device
                    );
                    */
                }
                h.to_device(target_device)?
            };

            // Pass KV Cache and Position
            let w_state = &w_states[i];
            let cache = &mut self.kv_caches[i];
            let pos = self.current_pos;

            let (h_new, w_new) = layer.forward(&h_layer, w_state, cache, pos)?;

            w_states[i] = w_new;
            h = h_new;
        }

        // [Hybrid Fix] Ensure input to Final Norm is on the correct device
        let norm_device = self.norm.weight.device();
        let h = if h.device().same_device(norm_device) {
            h
        } else {
            h.to_device(norm_device)?
        };

        let h_norm = self.norm.forward(&h)?;

        // [Hybrid Fix] Ensure input to lm_head is on correct device (may be CPU when lm_head_cpu=true)
        let lm_head_device = self.lm_head.weight().device();
        let h_norm = if h_norm.device().same_device(lm_head_device) {
            h_norm
        } else {
            h_norm.to_device(lm_head_device)?
        };

        let logits = self.lm_head.forward(&h_norm)?;

        // Advance Position
        self.current_pos += 1;

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
            let w_state = &w_states[i];
            // Chunkwise usually implies TTT or specific training mode.
            // Attention implementation of chunkwise is limited in block.rs
            let (h_new, w_new) = layer.forward_chunkwise(&h, w_state, chunk_size)?;
            w_states[i] = w_new;
            h = h_new;
        }

        // [Hybrid Fix] Ensure input to Final Norm is on the correct device
        let norm_device = self.norm.weight.device();
        let h = if h.device().same_device(norm_device) {
            h
        } else {
            h.to_device(norm_device)?
        };

        let h_norm = self.norm.forward(&h)?;

        // [Hybrid Fix] Ensure input to lm_head is on correct device (may be CPU when lm_head_cpu=true)
        let lm_head_device = self.lm_head.weight().device();
        let h_norm = if h_norm.device().same_device(lm_head_device) {
            h_norm
        } else {
            h_norm.to_device(lm_head_device)?
        };

        let logits = self.lm_head.forward(&h_norm)?;
        Ok(logits)
    }

    /// Helper for Python to check weights
    pub fn collect_tensors(&self) -> std::collections::HashMap<String, Tensor> {
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "embed.weight".to_string(),
            self.embedding.embeddings().clone(),
        );

        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);
            tensors.insert(
                format!("{}.norm1.weight", prefix),
                layer.norm1.weight.clone(),
            );

            // Helper to extract weight
            let get_weight = |l: &crate::layers::AdaptiveBitLinear| -> Option<Tensor> {
                if let Some(legacy) = &l.legacy_linear {
                    Some(legacy.weight.clone())
                } else {
                    l.reconstructed_weight.clone() // Return reconstructed weight if legacy not found
                }
            };

            match &layer.core {
                crate::model::block::LayerDispatch::TTT(ttt) => {
                    if let Some(w) = get_weight(&ttt.proj_down) {
                        tensors.insert(format!("{}.ttt.down.weight", prefix), w);
                    }
                    if let Some(w) = get_weight(&ttt.proj_up) {
                        tensors.insert(format!("{}.ttt.up.weight", prefix), w);
                    }
                }
                crate::model::block::LayerDispatch::Attention(attn) => {
                    if let Some(w) = get_weight(&attn.q_proj) {
                        tensors.insert(format!("{}.self_attn.q_proj.weight", prefix), w);
                    }
                    if let Some(w) = get_weight(&attn.k_proj) {
                        tensors.insert(format!("{}.self_attn.k_proj.weight", prefix), w);
                    }
                    if let Some(w) = get_weight(&attn.v_proj) {
                        tensors.insert(format!("{}.self_attn.v_proj.weight", prefix), w);
                    }
                    if let Some(w) = get_weight(&attn.o_proj) {
                        tensors.insert(format!("{}.self_attn.o_proj.weight", prefix), w);
                    }
                }
            }

            tensors.insert(
                format!("{}.norm2.weight", prefix),
                layer.norm2.weight.clone(),
            );

            if let Some(w) = get_weight(&layer.mlp.w1) {
                tensors.insert(format!("{}.mlp.gate_proj.weight", prefix), w);
            }
            if let Some(w) = get_weight(&layer.mlp.w2) {
                tensors.insert(format!("{}.mlp.down_proj.weight", prefix), w);
            }
            if let Some(w) = get_weight(&layer.mlp.w3) {
                tensors.insert(format!("{}.mlp.up_proj.weight", prefix), w);
            }
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
    /// Accumulated experience (Token Count) - "Soul Level"
    pub soul_level: u64,
}

impl Llama {
    pub fn load<P: AsRef<Path>>(
        model_path: P,
        tokenizer_path: P,
        config: BitLlamaConfig,
    ) -> Result<Self> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

        // Load Tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(candle_core::Error::wrap)?;

        // Lock File (ensure exclusive access if training, shared if inference)
        // For simplicity, just open standard file.
        let file = std::fs::File::open(&model_path)?;
        // fs2::FileExt::lock_shared(&file)?; // Optional: file locking

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };

        let model = BitLlama::load(config, vb)?;
        let w_states = model.new_w_states();

        Ok(Self {
            model,
            tokenizer,
            device,
            w_states,
            _lock_file: Some(file),
            soul_level: 0,
        })
    }

    /// Load model automatically from directory (or file path)
    pub fn load_auto<P: AsRef<Path>>(input_path: P) -> Result<Self> {
        let path = input_path.as_ref();
        let dir = if path.is_file() {
            path.parent().unwrap_or(path)
        } else {
            path
        };

        let config_path = dir.join("config.json");
        let tokenizer_path = dir.join("tokenizer.json");

        // Find safetensors
        let mut model_path = dir.join("model.safetensors");
        if !model_path.exists() {
            // Check for weight.safetensors or others
            model_path = dir.join("weight.safetensors");
            if !model_path.exists() {
                candle_core::bail!("No model.safetensors found in {:?}", dir);
            }
        }

        // Load Config
        let config_str = std::fs::read_to_string(&config_path).map_err(candle_core::Error::wrap)?;
        let config: BitLlamaConfig =
            serde_json::from_str(&config_str).map_err(candle_core::Error::wrap)?;

        Self::load(model_path, tokenizer_path, config)
    }

    pub fn reset_state(&mut self) -> Result<()> {
        self.model.reset_kv_cache();
        self.soul_level = 0;
        // Reset/Re-init TTT w_states
        let device = self.device.clone();
        let dim = self.model.config.hidden_dim;
        self.w_states =
            vec![Tensor::zeros((dim, dim), DType::F32, &device)?; self.model.layers.len()];
        Ok(())
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let callback = |_token: &str| Ok(true);
        self.stream_completion(prompt, max_tokens, 0.8, callback)
    }

    pub fn stream_completion<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temp: f64,
        mut callback: F,
    ) -> Result<String>
    where
        F: FnMut(&str) -> anyhow::Result<bool>, // using anyhow for flexible callback error
    {
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(candle_core::Error::wrap)?;
        let mut token_ids = tokens.get_ids().to_vec();

        let mut output_str = String::from(prompt);

        // 1. Prefill
        for &id in &token_ids {
            let input = Tensor::new(&[id], &self.device)?.unsqueeze(0)?;
            let _ = self.model.forward_one(&input, &mut self.w_states)?;
        }

        // 2. Generate
        let mut last_token = *token_ids.last().unwrap();
        for _ in 0..max_tokens {
            let input = Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward_one(&input, &mut self.w_states)?;

            // Sampling with Temp
            let logits_v: Vec<f32> = logits.squeeze(0)?.squeeze(0)?.to_vec1()?;
            let next_token = if temp < TEMP_MIN {
                // Greedy
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i as u32)
                    .unwrap()
            } else {
                // Multinomial (Simple implementation or use rand/candle-nn sampler)
                // For now, let's stick to Greedy-ish or simple Softmax
                let _prs = candle_nn::ops::softmax(&logits.squeeze(0)?.squeeze(0)?, 0)?;
                // Mock sampling or just Greedy for now for stability
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };

            token_ids.push(next_token);
            last_token = next_token;

            // Decode
            let decoded = self
                .tokenizer
                .decode(&[next_token], true)
                .map_err(candle_core::Error::wrap)?;

            // Callback
            if !callback(&decoded).map_err(|e| candle_core::Error::Msg(e.to_string()))? {
                break;
            }
            output_str.push_str(&decoded);

            self.soul_level += 1;

            if next_token == 2 {
                // EOS
                break;
            }
        }
        Ok(output_str)
    }

    // TTT Training Update (Learn)
    pub fn learn(&mut self, text: &str) -> Result<()> {
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(candle_core::Error::wrap)?;
        let token_ids = tokens.get_ids().to_vec();

        // Chunkwise Forward (Naive one by one for now to share state logic)
        // Ideally use forward_chunkwise for speed.

        // Simple forward pass to update w_states
        for &id in &token_ids {
            let input = Tensor::new(&[id], &self.device)?.unsqueeze(0)?;
            let _ = self.model.forward_one(&input, &mut self.w_states)?;
            self.soul_level += 1;
        }
        Ok(())
    }

    // Memory Persistence
    pub fn save_memory<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let w_tensors: std::collections::HashMap<String, Tensor> = self
            .w_states
            .iter()
            .enumerate()
            .map(|(i, t)| (format!("layer_{}", i), t.clone()))
            .collect();
        // Also save soul_level
        // .safetensors doesn't support metadata easily in save helper?
        // Just save tensors.
        // Or inject a scalar tensor for soul level.
        // let sl = Tensor::from_vec(vec![self.soul_level as f32], (1,), &self.device)?;
        // w_tensors.insert("soul_level".to_string(), sl);

        candle_core::safetensors::save(&w_tensors, path)?;
        Ok(())
    }

    pub fn load_memory<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &self.device)? };

        for i in 0..self.w_states.len() {
            if let Ok(t) = vb.get(
                (self.model.config.hidden_dim, self.model.config.hidden_dim),
                &format!("layer_{}", i),
            ) {
                self.w_states[i] = t;
            }
        }
        // Restore Soul Level if present
        // if let Ok(sl) = vb.get((1,), "soul_level") {
        //     let v: Vec<f32> = sl.to_vec1()?;
        //     self.soul_level = v[0] as u64;
        // }

        Ok(())
    }
}
