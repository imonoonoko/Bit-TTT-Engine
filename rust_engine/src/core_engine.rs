use candle_core::{DType, Module, Result, Tensor};
use candle_nn::VarBuilder;
use pyo3::prelude::*;

// --- RMSNorm ---
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
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
            return x.matmul(w_t);
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

        // Matmul that handles (B, T, D) or (B, D) or (D)
        // Linear: x @ w.T
        x.matmul(&w_ste.t()?)
    }
}

// --- MLP (SwiGLU) ---
pub struct SwiGLU {
    w1: BitLinear, // Gate
    w2: BitLinear, // Down
    w3: BitLinear, // Up
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
        let norm = norm.broadcast_add(&Tensor::new(&[1e-6f32], x_t.device())?)?;
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
}

// --- BitLlama Block ---
pub struct BitLlamaBlock {
    norm1: RMSNorm,
    ttt: TTTLayer,
    norm2: RMSNorm,
    mlp: SwiGLU,
}

impl BitLlamaBlock {
    pub fn load(dim: usize, inner_lr: f64, vb: VarBuilder) -> Result<Self> {
        let norm1 = RMSNorm::load(dim, 1e-5, vb.pp("norm1"))?;
        let ttt = TTTLayer::load(dim, inner_lr, vb.pp("ttt"))?;
        let norm2 = RMSNorm::load(dim, 1e-5, vb.pp("norm2"))?;
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

#[pyclass]
#[derive(Clone, Copy)]
pub struct BitLlamaConfig {
    #[pyo3(get, set)]
    pub vocab_size: usize,
    #[pyo3(get, set)]
    pub hidden_dim: usize,
    #[pyo3(get, set)]
    pub num_layers: usize,
    #[pyo3(get, set)]
    pub inner_lr: f64,
}

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

impl BitLlama {
    pub fn load(cfg: BitLlamaConfig, vb: VarBuilder) -> Result<Self> {
        let embedding = candle_nn::embedding(cfg.vocab_size, cfg.hidden_dim, vb.pp("embed"))?;

        let mut layers = Vec::new();
        for i in 0..cfg.num_layers {
            let layer = BitLlamaBlock::load(
                cfg.hidden_dim,
                cfg.inner_lr,
                vb.pp(&format!("layers.{}", i)),
            )?;
            layers.push(layer);
        }

        let norm = RMSNorm::load(cfg.hidden_dim, 1e-5, vb.pp("norm_f"))?;
        let lm_head = candle_nn::linear(cfg.hidden_dim, cfg.vocab_size, vb.pp("lm_head"))?;

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
    pub fn forward_one(&self, x: &Tensor, w_states: &mut Vec<Tensor>) -> Result<Tensor> {
        let mut h = self.embedding.forward(x)?.squeeze(0)?; // (B, D) or (D)

        for (i, layer) in self.layers.iter().enumerate() {
            let (h_new, w_new) = layer.forward(&h, &w_states[i])?;
            h = h_new;
            w_states[i] = w_new;
        }

        let h_norm = self.norm.forward(&h)?;
        let logits = self.lm_head.forward(&h_norm)?;
        Ok(logits)
    }
}
