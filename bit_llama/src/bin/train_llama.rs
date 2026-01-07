use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, Optimizer, VarBuilder, VarMap};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

// Import core engine
#[path = "../core_engine.rs"]
mod core_engine;
use core_engine::{BitLlama, BitLlamaConfig};

const CONTEXT_LEN: usize = 128; // Reduced for speed (Loop overhead)
const BATCH_SIZE: usize = 32; // Optimized for 8GB VRAM (Sweet spot?)
const DIM: usize = 256;
const LAYERS: usize = 4;
const VOCAB: usize = 16384; // Matches our BPE
const LR: f64 = 0.001;

// Data Loader
struct DataLoader {
    data: Vec<u16>,
    cursor: usize,
}

impl DataLoader {
    fn new(path: &str) -> Result<Self> {
        let f = File::open(path)?;
        let mut reader = BufReader::new(f);
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer)?;

        // Convert u8 buffer to u16
        // Assumes Little Endian (from numpy tobytes)
        let data: Vec<u16> = buffer
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();

        Ok(Self { data, cursor: 0 })
    }

    fn next_batch(
        &mut self,
        batch_size: usize,
        len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for _ in 0..batch_size {
            if self.cursor + len + 1 >= self.data.len() {
                self.cursor = 0; // Reset
            }
            let chunk = &self.data[self.cursor..self.cursor + len + 1];
            inputs.extend_from_slice(&chunk[0..len]);
            targets.extend_from_slice(&chunk[1..len + 1]);
            self.cursor += len;
        }

        // Fix: Candle doesn't support u16 tensors. Convert to u32.
        let inputs_u32: Vec<u32> = inputs.iter().map(|&x| x as u32).collect();
        let targets_u32: Vec<u32> = targets.iter().map(|&x| x as u32).collect();

        let inp_tensor = Tensor::from_slice(&inputs_u32, (batch_size, len), device)?;
        let tgt_tensor = Tensor::from_slice(&targets_u32, (batch_size, len), device)?;

        Ok((inp_tensor, tgt_tensor))
    }
}

fn main() -> Result<()> {
    println!("--- Bit-Llama Training (TinyStories) ---");
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);

    // 1. Data
    let data_path = "../data/TinyStories/train.bin";
    if !Path::new(data_path).exists() {
        anyhow::bail!("Data not found: {}", data_path);
    }
    let mut loader = DataLoader::new(data_path)?;
    println!("Data Loaded. Total tokens: {}", loader.data.len());

    // 2. Model
    let config = BitLlamaConfig {
        vocab_size: VOCAB,
        hidden_dim: DIM,
        num_layers: LAYERS,
        inner_lr: 0.1,
    };

    let mut varmap = VarMap::new();
    if Path::new("bit_llama_checkpoint.safetensors").exists() {
        println!("Resuming from checkpoint...");
        varmap.load("bit_llama_checkpoint.safetensors")?;
    }
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = BitLlama::load(config, vb)?;

    let params = candle_nn::ParamsAdamW {
        lr: LR,
        ..Default::default()
    };
    let mut adam = candle_nn::AdamW::new(varmap.all_vars(), params)?;

    // 3. Loop
    let steps = 1000; // Demo run
    let log_interval = 10;

    println!("Starting Training Loop ({} steps)...", steps);

    for step in 0..steps {
        let (inputs, targets) = loader.next_batch(BATCH_SIZE, CONTEXT_LEN, &device)?;
        // Shape: (B, T)

        // Reset Fast Weights for Batch
        // Layers x (B, d, d)
        let d_small = DIM / 4;
        let mut w_states = Vec::new();
        for _ in 0..LAYERS {
            w_states.push(Tensor::zeros(
                (BATCH_SIZE, d_small, d_small),
                DType::F32,
                &device,
            )?);
        }

        let mut loss_step = Tensor::new(0.0f32, &device)?;

        // Recursive Forward (Token by Token)
        // Optimization: In real Llama, we use parallel computation.
        // Here, we loop T. It's slow in Rust Debug, acceptable in Release?

        let _batch_loss_accum = 0.0;

        // Input: (B, T). We iterate t.
        for t in 0..CONTEXT_LEN {
            let input_col = inputs.narrow(1, t, 1)?.squeeze(1)?.contiguous()?; // (B)
            let target_col = targets.narrow(1, t, 1)?.squeeze(1)?.contiguous()?; // (B)

            let mut h = model.embedding.forward(&input_col)?; // (B, D)

            // Pass through layers
            for (l_idx, layer) in model.layers.iter().enumerate() {
                let (h_new, w_new) = layer.forward(&h, &w_states[l_idx])?;
                h = h_new;
                w_states[l_idx] = w_new;
            }

            let h_norm = model.norm.forward(&h)?;
            let logits = model.lm_head.forward(&h_norm)?; // (B, V)

            let loss = candle_nn::loss::cross_entropy(&logits, &target_col)?;
            loss_step = (loss_step + loss)?;
        }

        // Backprop on Total Loss / T
        let loss_scaled = (loss_step / (CONTEXT_LEN as f64))?;
        adam.backward_step(&loss_scaled)?;

        if step % log_interval == 0 {
            let val = loss_scaled.to_scalar::<f32>()?;
            println!("Step {:4} | Loss: {:.4} (Saved)", step, val);
            varmap.save("bit_llama_checkpoint.safetensors")?;
        }
    }

    println!("Saving Bit-Llama...");
    varmap.save("bit_llama_v1.safetensors")?;

    Ok(())
}
