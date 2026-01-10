//! Core Engine for Bit-Llama Model
//!
//! This module contains the primary implementation of the Bit-Llama architecture:
//! - BitLinear (1.58-bit quantized linear layer)
//! - SwiGLU (MLP activation)
//! - TTTLayer (Test-Time Training with online learning)
//! - BitLlamaBlock and BitLlama (full model)

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use std::path::Path;
use tokenizers::Tokenizer;

// ============================================================
// Constants
// ============================================================
/// Epsilon for RMSNorm numerical stability.
const RMS_NORM_EPS: f64 = 1e-5;

/// Epsilon for TTT layer normalization.
const TTT_NORM_EPS: f32 = 1e-6;

/// Minimum temperature for sampling to prevent division by zero.
const TEMP_MIN: f64 = 1e-6;

// --- Helper Trait for Robust Operations ---
trait TensorExt {
    fn matmul_robust(&self, rhs: &Tensor) -> Result<Tensor>;
}

impl TensorExt for Tensor {
    fn matmul_robust(&self, rhs: &Tensor) -> Result<Tensor> {
        let lhs = self.contiguous()?;
        let rhs = rhs.contiguous()?;
        let lhs_rank = lhs.rank();

        if lhs_rank == 1 {
            // [D] @ [D, Out] -> [Out]
            lhs.unsqueeze(0)?.matmul(&rhs)?.squeeze(0)
        } else if lhs_rank == 2 {
            lhs.matmul(&rhs)
        } else {
            // [B, T, D] @ [D, Out] -> [B, T, Out]
            // Flatten to [B*T, D]
            // Generic Flatten:
            let flattened = lhs.flatten(0, lhs_rank - 2)?; // [Batch*, D]
            let out = flattened.matmul(&rhs)?; // [Batch*, Out]

            // Reshape back
            let mut new_shape = lhs.dims()[..lhs_rank - 1].to_vec();
            new_shape.push(out.dim(1)?);
            out.reshape(new_shape)
        }
    }
}

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::{pyclass, pymethods};

// --- RMSNorm ---
pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl RMSNorm {
    pub fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight =
            vb.get_with_hints((dim,), "weight", candle_nn::init::DEFAULT_KAIMING_NORMAL)?; // or ONES
                                                                                           // Actually RMSNorm usually init with Ones.
                                                                                           // But VarBuilder defaults are fine if we use `get`.
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = DType::F32; // Always use F32 for accumulation
        let x_f32 = x.to_dtype(internal_dtype)?;
        let dim = x_f32.rank() - 1;
        let hidden_size = x_f32.dim(dim)?;

        let norm_x = (x_f32.sqr()?.sum_keepdim(dim)? / (hidden_size as f64))?;
        let x_normed = x_f32.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;

        let weight = self
            .weight
            .to_dtype(internal_dtype)?
            .broadcast_as(x_normed.shape())?;
        let result = (x_normed * weight)?;

        result.to_dtype(x_dtype)
    }
}

// --- BitLinear (Candle Implementation) ---
pub struct BitLinear {
    pub weight: Tensor,
    #[allow(dead_code)]
    pub in_features: usize,
    #[allow(dead_code)]
    pub out_features: usize,
    // [Optimization] Pre-computed weights for inference (W_quant.T)
    pub inference_params: Option<Tensor>,
}

impl BitLinear {
    pub fn load(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let init = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let weight = vb.get_with_hints((out_dim, in_dim), "weight", init)?;
        Ok(Self {
            weight,
            in_features: in_dim,
            out_features: out_dim,
            inference_params: None,
        })
    }

    pub fn precompute_for_inference(&mut self) -> Result<()> {
        let w = &self.weight;
        // 1.58-bit Quantization logic (Same as forward)
        let scale = w.abs()?.mean_all()?;
        let w_scaled = (w / scale.to_scalar::<f32>()? as f64)?;
        let w_quant = w_scaled.round()?.clamp(-1.0, 1.0)?;

        // Transpose for matmul: x @ w.T
        // Storing w.T directly saves transpose op at runtime
        let w_quant_t = w_quant.t()?.detach();

        self.inference_params = Some(w_quant_t);
        Ok(())
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // [Optimization] Fast path for inference
        if let Some(w_t) = &self.inference_params {
            // Uses robust matmul to support 1D inputs
            return x.matmul_robust(w_t);
        }

        let w = &self.weight;
        // 1.58-bit Quantization
        let scale = w.abs()?.mean_all()?;
        let w_scaled = (w / scale.to_scalar::<f32>()? as f64)?;
        let w_quant = w_scaled.round()?.clamp(-1.0, 1.0)?;

        // STE
        let diff = (w_quant - &w_scaled)?;
        let detached_diff = diff.detach();
        let w_ste = (detached_diff + &w_scaled)?;

        // Linear: x @ w.T
        x.matmul_robust(&w_ste.t()?)
    }
}

// --- MLP (SwiGLU) ---
pub struct SwiGLU {
    pub w1: BitLinear, // Gate
    pub w2: BitLinear, // Down
    pub w3: BitLinear, // Up
}

impl SwiGLU {
    pub fn load(hidden_dim: usize, intermediate_dim: usize, vb: VarBuilder) -> Result<Self> {
        let w1 = BitLinear::load(hidden_dim, intermediate_dim, vb.pp("gate_proj"))?;
        let w2 = BitLinear::load(intermediate_dim, hidden_dim, vb.pp("down_proj"))?;
        let w3 = BitLinear::load(hidden_dim, intermediate_dim, vb.pp("up_proj"))?;
        Ok(Self { w1, w2, w3 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_gate = self.w1.forward(x)?;
        let x_up = self.w3.forward(x)?;
        // SwiGLU = (Results of gate * SiLU) * up
        // Wait, standard is: down(silu(gate) * up)
        let silu_gate = candle_nn::ops::silu(&x_gate)?;
        let hidden = (silu_gate * x_up)?;
        self.w2.forward(&hidden)
    }

    pub fn precompute_for_inference(&mut self) -> Result<()> {
        self.w1.precompute_for_inference()?;
        self.w2.precompute_for_inference()?;
        self.w3.precompute_for_inference()?;
        Ok(())
    }
}

// --- TTTLayer (Updated for Batching) ---
pub struct TTTLayer {
    #[allow(dead_code)]
    pub hidden_dim: usize,
    #[allow(dead_code)]
    pub d_small: usize,
    pub proj_down: BitLinear,
    pub proj_up: BitLinear,
    pub inner_lr: f64,
}

impl TTTLayer {
    pub fn load(hidden_dim: usize, inner_lr: f64, vb: VarBuilder) -> Result<Self> {
        let d_small = hidden_dim / 4;
        Ok(Self {
            hidden_dim,
            d_small,
            proj_down: BitLinear::load(hidden_dim, d_small, vb.pp("down"))?,
            proj_up: BitLinear::load(d_small, hidden_dim, vb.pp("up"))?,
            inner_lr,
        })
    }

    pub fn precompute_for_inference(&mut self) -> Result<()> {
        self.proj_down.precompute_for_inference()?;
        self.proj_up.precompute_for_inference()?;
        Ok(())
    }

    // Handles Batching: w_state is (B, D_small, D_small) or (D_small, D_small)
    // x is (B, Hidden) or (Hidden)
    pub fn forward_update(&self, w_state: &Tensor, x_t: &Tensor) -> Result<(Tensor, Tensor)> {
        // 1. Project Down
        let feat = self.proj_down.forward(x_t)?;

        // Normalize (Simple L2 per vector)
        // Dim is last
        let last_dim = feat.rank() - 1;
        let norm = feat.sqr()?.sum_keepdim(last_dim)?.sqrt()?;
        // Avoid div by zero
        let norm = norm.broadcast_add(&Tensor::new(&[TTT_NORM_EPS], x_t.device())?)?;
        let feat_norm = feat.broadcast_div(&norm)?;

        // 2. Predict (TTT)
        // feat_norm: (B, d) -> (B, d, 1)
        // w_state: (B, d, d)
        // pred = w @ feat

        let feat_expanded = feat_norm.unsqueeze(last_dim + 1)?;
        let pred_inner = w_state.matmul(&feat_expanded)?.squeeze(last_dim + 1)?;

        // 3. Loss & Grad
        let diff = (&pred_inner - &feat_norm)?;
        let _loss = diff.sqr()?.sum_all()?; // Optional return

        // Grad = diff * feat.T
        // diff: (B, d, 1)
        // feat: (B, 1, d) (transpose of (B, d, 1))
        let diff_ed = diff.unsqueeze(last_dim + 1)?;
        // feat_norm: (B, d). unsqueeze(1) -> (B, 1, d). No transpose needed.
        let feat_ed_t = feat_norm.unsqueeze(last_dim)?;

        // Outer Product: (B, d, 1) @ (B, 1, d) -> (B, d, d)
        let grad = diff_ed.matmul(&feat_ed_t)?;

        // 4. Update
        let w_new = (w_state - grad * self.inner_lr)?.detach();

        // 5. Project Up (Residual Logic)
        let out_feat = self.proj_up.forward(&pred_inner)?;

        Ok((out_feat, w_new))
    }

    // --- Parallel Chunkwise Implementation ---
    // x: (B, T, Hidden)
    // w_state: (B, D_small, D_small) aka w_init (usually zeros)
    // Returns: (output: (B, T, Hidden), w_final: (B, D_small, D_small))
    pub fn forward_chunkwise(
        &self,
        w_state: &Tensor,
        x: &Tensor,
        chunk_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        // 1. Project Down (All tokens at once) -> (B, T, D_small)
        let feat = self.proj_down.forward(x)?;

        // Normalize
        let norm = feat.sqr()?.sum_keepdim(2)?.sqrt()?;
        let norm = norm.broadcast_add(&Tensor::new(&[TTT_NORM_EPS], x.device())?)?;
        let feat_norm = feat.broadcast_div(&norm)?; // (B, T, D_small)

        let (_b_sz, t_len, _d_small) = feat_norm.dims3()?;
        let mut current_w = w_state.clone();
        let mut outputs = Vec::new();

        // 2. Chunk Loop
        // We iterate over chunks of size `chunk_size`
        // Using chunks means:
        //   - Z_chunk = W_curr @ X_chunk^T  Matrix-Matrix (B, D, D) @ (B, D, C) -> (B, D, C)
        //   - Grad = (Z - X) @ X^T          Matrix-Matrix (B, D, C) @ (B, C, D) -> (B, D, D)
        //   - W_next = W_curr - lr * Grad

        // Ensure we handle T not divisible by chunk_size if necessary (input.chunk handles this)
        // But manual iteration is safer for reshaping logic.
        let num_chunks = t_len.div_ceil(chunk_size);

        for i in 0..num_chunks {
            let start = i * chunk_size;
            let len = std::cmp::min(chunk_size, t_len - start);

            // Get Chunk: (B, C, D_small)
            let x_chunk = feat_norm.narrow(1, start, len)?;

            // Prepare for Matrix Mul: (B, D, C)
            // x_chunk_t: (B, D, C)
            let x_chunk_t = x_chunk.transpose(1, 2)?;

            // Forward: Z = W @ X^T
            // (B, D, D) @ (B, D, C) = (B, D, C)
            let z_chunk_t = current_w.matmul(&x_chunk_t)?;

            // Transpose back for diff: (B, C, D)
            let z_chunk = z_chunk_t.transpose(1, 2)?;

            // Grad: G = (Z - X) @ X or similar?
            // Diff: (B, C, D)
            let diff = (&z_chunk - &x_chunk)?;

            // Update Rule: W_next = W - eta * (z-x) @ x^T
            // diff: (B, C, D). x_chunk: (B, C, D)
            // We want (B, D, D) update.
            // diff.T @ x_chunk ?? No.
            // Update is sum of outer products: sum_t (z_t - x_t) * x_t^T
            // Matrix form: (Z_chunk - X_chunk)^T @ X_chunk ??
            // (B, D, C) @ (B, C, D) -> (B, D, D). Correct.
            // diff is (B, C, D). diff_t is (B, D, C).
            let diff_t = diff.transpose(1, 2)?;
            let grad = diff_t.matmul(&x_chunk)?; // (B, D, D)

            // Update W
            current_w = (current_w - grad * self.inner_lr)?;

            // Store Z_chunk as output (before project up)
            outputs.push(z_chunk);
        }

        // 3. Concat Outputs -> (B, T, D_small)
        let pred_all = Tensor::cat(&outputs, 1)?;

        // 4. Project Up -> (B, T, Hidden)
        let out_feat = self.proj_up.forward(&pred_all)?;

        Ok((out_feat, current_w))
    }
}

// --- BitLlama Block ---
pub struct BitLlamaBlock {
    pub norm1: RMSNorm,
    pub ttt: TTTLayer,
    pub norm2: RMSNorm,
    pub mlp: SwiGLU,
}

impl BitLlamaBlock {
    pub fn load(dim: usize, inner_lr: f64, vb: VarBuilder) -> Result<Self> {
        let norm1 = RMSNorm::load(dim, RMS_NORM_EPS, vb.pp("norm1"))?;
        let ttt = TTTLayer::load(dim, inner_lr, vb.pp("ttt"))?;
        let norm2 = RMSNorm::load(dim, RMS_NORM_EPS, vb.pp("norm2"))?;
        // MLP Dim: Usually 4 * Dim, or 8/3 * Dim (SwiGLU convention)
        // Let's use 2.5 * Dim (BitNet uses lighter MLP sometimes) or 4 * Dim.
        // TinyStories is small. 4 x Dim is good.
        let mlp_dim = dim * 4;
        let mlp = SwiGLU::load(dim, mlp_dim, vb.pp("mlp"))?;

        Ok(Self {
            norm1,
            ttt,
            norm2,
            mlp,
        })
    }

    pub fn precompute_for_inference(&mut self) -> Result<()> {
        self.ttt.precompute_for_inference()?;
        self.mlp.precompute_for_inference()?;
        Ok(())
    }

    pub fn forward(&self, x: &Tensor, w_state: &Tensor) -> Result<(Tensor, Tensor)> {
        // 1. TTT Branch
        let residual = x;
        let x_norm = self.norm1.forward(x)?;
        let (ttt_out, w_new) = self.ttt.forward_update(w_state, &x_norm)?;
        let x_mid = (residual + ttt_out)?;

        // 2. MLP Branch
        let residual = &x_mid;
        let x_norm2 = self.norm2.forward(&x_mid)?;
        let mlp_out = self.mlp.forward(&x_norm2)?;
        let x_out = (residual + mlp_out)?;

        Ok((x_out, w_new))
    }

    pub fn forward_chunkwise(
        &self,
        x: &Tensor,
        w_state: &Tensor,
        chunk_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        // 1. TTT Branch (Chunkwise)
        let residual = x;
        let x_norm = self.norm1.forward(x)?;
        let (ttt_out, w_final) = self.ttt.forward_chunkwise(w_state, &x_norm, chunk_size)?;
        let x_mid = (residual + ttt_out)?;

        // 2. MLP Branch (Standard Batch Linear)
        // MLP matches input shape (B, T, D) -> (B, T, D)
        let residual = &x_mid;
        let x_norm2 = self.norm2.forward(&x_mid)?;
        let mlp_out = self.mlp.forward(&x_norm2)?;
        let x_out = (residual + mlp_out)?;

        Ok((x_out, w_final))
    }
}

// --- BitLlama Model ---
pub struct BitLlama {
    pub embedding: candle_nn::Embedding,
    pub layers: Vec<BitLlamaBlock>,
    pub norm: RMSNorm,
    pub lm_head: candle_nn::Linear,
    #[allow(dead_code)]
    pub config: BitLlamaConfig,
}

// --- BitLlamaConfig (Python Version) ---
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

// --- BitLlamaConfig (Rust Version) ---
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

impl BitLlama {
    pub fn load(cfg: BitLlamaConfig, vb: VarBuilder) -> Result<Self> {
        let embedding = candle_nn::embedding(cfg.vocab_size, cfg.hidden_dim, vb.pp("embed"))?;

        let mut layers = Vec::new();
        for i in 0..cfg.num_layers {
            let layer =
                BitLlamaBlock::load(cfg.hidden_dim, cfg.inner_lr, vb.pp(format!("layers.{}", i)))?;
            layers.push(layer);
        }

        let norm = RMSNorm::load(cfg.hidden_dim, RMS_NORM_EPS, vb.pp("norm_f"))?;
        let lm_head = candle_nn::linear_no_bias(cfg.hidden_dim, cfg.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            embedding,
            layers,
            norm,
            lm_head,
            config: cfg,
        })
    }

    pub fn precompute_for_inference(&mut self) -> Result<()> {
        for layer in self.layers.iter_mut() {
            layer.precompute_for_inference()?;
        }
        Ok(())
    }

    // Forward Step (Single Token) - used for inference
    #[allow(dead_code)]
    pub fn forward_one(&self, x: &Tensor, w_states: &mut [Tensor]) -> Result<Tensor> {
        let mut h = self.embedding.forward(x)?.squeeze(0)?; // (B, D) or (D)

        for (i, layer) in self.layers.iter().enumerate() {
            let (h_new, w_new) = layer.forward(&h, &w_states[i])?;
            h = h_new;
            w_states[i] = w_new;
        }

        let h_norm = self.norm.forward(&h)?;
        // Use robust matmul via TensorExt.
        // Need to simulate linear layer: x @ w.T + b
        // lm_head has no bias. weight is [Out, In].
        // x @ w.T
        // TensorExt is implemented on Tensor.
        // We need to access candle_nn::Linear weight.
        let w = self.lm_head.weight();
        // Since candle_nn::Linear::forward failed on 1D, we use our robust matmul.
        let logits = h_norm.matmul_robust(&w.t()?)?;
        Ok(logits)
    }

    // Forward Chunkwise (Parallel Training)
    // x: (B, T)
    // w_states: Initial states (usually zeros). Will be updated to final states?
    // Actually, for training, we usually discard final state or use it for next sequence.
    // Here we return logits (B, T, V).
    pub fn forward_chunkwise(
        &self,
        x: &Tensor,
        w_states: &mut [Tensor],
        chunk_size: usize,
    ) -> Result<Tensor> {
        // Embed: (B, T) -> (B, T, D)
        let mut h = self.embedding.forward(x)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let (h_new, w_final) = layer.forward_chunkwise(&h, &w_states[i], chunk_size)?;
            h = h_new;
            w_states[i] = w_final;
        }

        let h_norm = self.norm.forward(&h)?;

        // LM Head: (B, T, D) @ (V, D)^T -> (B, T, V)
        // Use matmul_robust to handle 3D input
        let w = self.lm_head.weight();
        let logits = h_norm.matmul_robust(&w.t()?)?;

        Ok(logits)
    }
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
            tensors.insert(
                format!("{}.ttt.down.weight", prefix),
                layer.ttt.proj_down.weight.clone(),
            );
            tensors.insert(
                format!("{}.ttt.up.weight", prefix),
                layer.ttt.proj_up.weight.clone(),
            );
            tensors.insert(
                format!("{}.norm2.weight", prefix),
                layer.norm2.weight.clone(),
            );
            tensors.insert(
                format!("{}.mlp.gate_proj.weight", prefix),
                layer.mlp.w1.weight.clone(),
            );
            tensors.insert(
                format!("{}.mlp.down_proj.weight", prefix),
                layer.mlp.w2.weight.clone(),
            );
            tensors.insert(
                format!("{}.mlp.up_proj.weight", prefix),
                layer.mlp.w3.weight.clone(),
            );
        }

        tensors.insert("norm_f.weight".to_string(), self.norm.weight.clone());
        tensors.insert("lm_head.weight".to_string(), self.lm_head.weight().clone());

        tensors
    }
}

// --- High-Level Rust API (Llama) ---

pub struct Llama {
    pub model: BitLlama,
    pub tokenizer: Tokenizer,
    pub device: candle_core::Device,
    pub w_states: Vec<Tensor>,
}

impl Llama {
    /// Auto-detects format and loads the model.
    /// Supports:
    /// - `.bitt` file (Fast Native Container)
    /// - Directory (Legacy: needs config.json + tokenizer.json + model.safetensors)
    /// - `.safetensors` file (Legacy: infers parent directory)
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
                    // Assume parent dir contains config/tokenizer
                    let parent = path.parent().unwrap_or(Path::new("."));
                    return Self::new_with_device(parent, device);
                }
            }
        } else if path.is_dir() {
            return Self::new_with_device(path, device);
        }

        // Fallback to trying as BITT if file, or error
        Self::from_bitt_file_with_device(path, device)
            .or_else(|_| Self::new_with_device(path, device))
            .map_err(|_| {
                anyhow::anyhow!(
                    "Failed to load model from {:?}. Not a valid .bitt file or legacy directory.",
                    path
                )
            })
    }

    /// Bit-TTT Native Container (.bitt) Loader
    /// Fast loading from a single file containing Config, Tokenizer, and Weights.
    pub fn from_bitt_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let device = candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);
        Self::from_bitt_file_with_device(path, &device)
    }

    pub fn from_bitt_file_with_device<P: AsRef<Path>>(
        path: P,
        device: &candle_core::Device,
    ) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)?;

        // Memmap for fast access
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // 1. Magic Check
        if mmap.len() < 12 || &mmap[0..4] != b"BITT" {
            anyhow::bail!("Invalid format: Not a .bitt file (Magic mismatch)");
        }

        // 2. Read Header Length (u64 LE) at offset 4
        // We can just read bytes directly
        let header_len_bytes: [u8; 8] = mmap[4..12].try_into()?;
        let header_len = u64::from_le_bytes(header_len_bytes) as usize;

        // 3. Parse Header (JSON)
        let header_start = 12;
        let header_end = header_start + header_len;
        if mmap.len() < header_end {
            anyhow::bail!("Invalid format: File too short for header");
        }

        let header_slice = &mmap[header_start..header_end];
        let header_json: serde_json::Value = serde_json::from_slice(header_slice)?;

        // Reconstruct Config
        let config: BitLlamaConfig = serde_json::from_value(header_json["config"].clone())
            .map_err(|e| anyhow::anyhow!("Failed to parse config from BITT header: {}", e))?;

        // Reconstruct Tokenizer
        use std::str::FromStr;
        let tokenizer_json_str = header_json["tokenizer"].to_string();
        let tokenizer = Tokenizer::from_str(&tokenizer_json_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse tokenizer from BITT header: {}", e))?;

        // 4. Load Weights (Safetensors Body)
        let body_start = header_end;
        let body_slice = &mmap[body_start..];

        // Create VarBuilder from buffer
        // Note: to_vec() incurs a copy.
        let vb = candle_nn::VarBuilder::from_buffered_safetensors(
            body_slice.to_vec(),
            DType::F32,
            device,
        )?;

        // 5. Build Model
        let mut model = BitLlama::load(config, vb)?;
        model.precompute_for_inference()?;

        // 6. Init State
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

        // 2. Load Config
        let config_path = dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| anyhow::anyhow!("Failed to read config.json: {}", e))?;
        let config: BitLlamaConfig = serde_json::from_str(&config_str)?;

        // 3. Load Tokenizer
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer.json: {}", e))?;

        // 4. Load Weights
        let weights_path = dir.join("model.safetensors");
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)?
        };

        // 5. Build Model
        let mut model = BitLlama::load(config, vb)?;
        model.precompute_for_inference()?;

        // 6. Init State
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
        })
    }

    // Streaming with Callback: fn(token_str) -> Result<Continue(bool)>
    // Returns full text at the end for convenience
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
        // Enforce temp > 0
        let temperature = if temp <= 0.0 { TEMP_MIN } else { temp };

        // 1. Tokenize
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!(e))?;
        let tokens = encoding.get_ids();

        if tokens.is_empty() {
            return Ok(String::new());
        }

        let mut all_tokens = tokens.to_vec();

        // 2. Prefill
        let mut next_token = *all_tokens.last().unwrap(); // Placeholder

        for &t in tokens {
            let input = Tensor::new(&[t], &self.device)?;
            let logits = self.model.forward_one(&input, &mut self.w_states)?;

            if t == *tokens.last().unwrap() {
                // Sample next token from last logits
                let logits_v = logits.squeeze(0)?;
                let prs = candle_nn::ops::softmax(&(&logits_v / temperature)?, 0)?;
                let prs_vec = prs.to_vec1::<f32>()?;
                next_token = Self::sample_multinomial(&prs_vec)?;
            }
        }

        all_tokens.push(next_token);

        // 3. Generation Loop with Cumulative Decoding
        // For byte-level BPE tokenizers, single-token decoding produces escape sequences.
        // We use cumulative decoding: decode(all_generated_tokens) and output the delta.
        let mut generated_tokens: Vec<u32> = vec![next_token];

        // Initialize prev_decoded - decode first generated token
        let mut prev_decoded = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!(e))?;

        // Output first token if not empty
        if !prev_decoded.is_empty() && !callback(&prev_decoded)? {
            let full_text = self
                .tokenizer
                .decode(&all_tokens, true)
                .map_err(|e| anyhow::anyhow!(e))?;
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

            // Cumulative decode: decode all generated tokens, output the new portion
            let current_decoded = self
                .tokenizer
                .decode(&generated_tokens, true)
                .map_err(|e| anyhow::anyhow!(e))?;

            // Calculate delta (new characters since last decode)
            let delta = if current_decoded.len() > prev_decoded.len() {
                &current_decoded[prev_decoded.len()..]
            } else {
                // Edge case: tokenizer may produce shorter output (unlikely but safe)
                &current_decoded
            };

            // Debug: Check if delta contains byte escapes and try to fix
            // The tokenizers library may return "\xNN" strings for raw byte tokens
            let processed_delta = if delta.contains("\\x") {
                // Try to interpret as escaped bytes
                // Pattern: \xNN where NN is hex
                let mut result = String::new();
                let mut chars = delta.chars().peekable();
                while let Some(c) = chars.next() {
                    if c == '\\' {
                        if chars.peek() == Some(&'x') {
                            chars.next(); // consume 'x'
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
            } else {
                delta.to_string()
            };

            if !processed_delta.is_empty() && !callback(&processed_delta)? {
                break;
            }
            prev_decoded = current_decoded;
        }

        let full_text = self
            .tokenizer
            .decode(&all_tokens, true)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(full_text)
    }

    pub fn export_to_bitt<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let path = path.as_ref();
        use std::io::Write;

        // 1. Collect Tensors
        let tensors = self.model.collect_tensors();

        // 2. Serialize Tensors to Buffer (Safetensors)
        // candle_core doesn't expose easy "save to buffer", but we can use safetensors crate directly if needed.
        // Or write to a temp file? writing to memory is better.
        // Actually, candle provides `save_to_file`. To save to buffer...
        // We can use `safetensors::tensor::serialize`. We need to convert Candle Tensor to View.
        // This is complex. Is there a simpler way?
        // candle::safetensors::save returns Result<()>.
        // Let's rely on `safetensors::SomeFunc`?
        // Wait, `candle` re-exports safetensors? No.
        // Let's just assume we can use a helper or write to a temporary buffer using `candle_core::pickle`? No.
        // Use `candle_core::safetensors::save` to a generic Writer? It only takes a Path.
        //
        // Workaround: We will use `safetensors::tensor::serialize` but that requires `safetensors` dependency with `candle` features.
        //
        // Easier: Write the safetensors part to a temporary file, read it back, then concat.
        // Or just implement the header writing manually and append the standard file.
        //
        // Strategy:
        // 1. Write Header to File.
        // 2. Append Safetensors body.
        // But `candle_core::safetensors::save` overwrites the file.
        //
        // Correct Strategy using available APIs:
        // 1. Write Safetensors to a temporary file.
        // 2. Read it all into memory (or stream copy).
        // 3. Construct Final File: Magic + Len + HeaderJSON + SafetensorsBody.

        // A. Create Safetensors Blob
        let temp_path = path.with_extension("temp.safetensors");
        candle_core::safetensors::save(&tensors, &temp_path)?;
        let safetensors_bytes = std::fs::read(&temp_path)?;
        let _ = std::fs::remove_file(temp_path); // Cleanup

        // B. Create Header
        let config_json = serde_json::to_value(self.model.config)?;
        // Tokenizer: we need the JSON string. `tokenizer.to_string(true)`?
        // method is `save(path, pretty)` or `to_string(pretty)`.
        let tokenizer_json_str = self
            .tokenizer
            .to_string(false)
            .map_err(|e| anyhow::anyhow!(e))?;

        let header = serde_json::json!({
            "config": config_json,
            "tokenizer": tokenizer_json_str
        });
        let header_vec = serde_json::to_vec(&header)?;
        let header_len = header_vec.len() as u64;

        // C. Write Final File
        let mut file = std::fs::File::create(path)?;

        // Magic
        file.write_all(b"BITT")?;

        // Header Len (u64 LE)
        file.write_all(&header_len.to_le_bytes())?;

        // Header
        file.write_all(&header_vec)?;

        // Body
        file.write_all(&safetensors_bytes)?;

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

// --- Python Bindings ---

#[cfg(feature = "python")]
#[pyclass(name = "BitLlama")]
pub struct PyBitLlama {
    inner: BitLlama,
    // Keep state in Python or Rust? Rust is safer for TTT.
    // For TTT, we need w_states.
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

        // Optimization: Pre-compute weights
        model
            .precompute_for_inference()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Initialize TTT state (w_state)
        // Shape: (1, d_small, d_small) for each layer
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

        // forward_one expects &mut [Tensor] for states
        let logits = self
            .inner
            .forward_one(&input, &mut self.w_states)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Convert logits to Vec<f32> for Python
        // logits: (VocabSize)
        let logits_vec = logits
            .squeeze(0)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(logits_vec)
    }
}
