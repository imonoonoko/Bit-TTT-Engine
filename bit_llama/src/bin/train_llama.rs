use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, Optimizer, VarBuilder, VarMap};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

// Import core engine
// #[path = "../core_engine.rs"]
// mod core_engine;
use cortex_rust::{BitLlama, BitLlamaConfig};

const CONTEXT_LEN: usize = 128; // Reduced for speed (Loop overhead)
const BATCH_SIZE: usize = 32; // Optimized for 8GB VRAM (Sweet spot?)
const DIM: usize = 256;
const LAYERS: usize = 4;
const VOCAB: usize = 16384; // Matches our BPE

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

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "0.001")]
    lr: f64,

    #[arg(long, default_value = "10000")]
    steps: usize,

    #[arg(long, default_value = "data/TinyStories/train.bin")]
    data: String,

    #[arg(long, default_value = "500")]
    save_interval: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("--- Bit-Llama Training (TinyStories) ---");
    println!(
        "Config: LR={}, Steps={}, Data={}",
        args.lr, args.steps, args.data
    );

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);

    // 1. Data (with fallback for different working directories)
    let mut data_path = args.data.clone();
    if !Path::new(&data_path).exists() {
        // Fallback: try prefixing with bit_llama/ (if running from Bit-TTT/)
        let fallback = format!("bit_llama/{}", &args.data);
        if Path::new(&fallback).exists() {
            data_path = fallback;
            println!("Using fallback data path: {}", data_path);
        } else {
            anyhow::bail!("Data not found at '{}' or '{}'", args.data, fallback);
        }
    }
    let mut loader = DataLoader::new(&data_path)?;
    println!("Data Loaded. Total tokens: {}", loader.data.len());

    // 2. Model
    let config = BitLlamaConfig {
        vocab_size: VOCAB,
        hidden_dim: DIM,
        num_layers: LAYERS,
        inner_lr: 0.1,
    };

    let mut varmap = VarMap::new();
    // 1. Init Model first (populates varmap with random weights)
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = BitLlama::load(config, vb)?;

    // 2. Overwrite with Checkpoint if exists (with fallback for different working directories)
    let checkpoint_path = if Path::new("bit_llama_checkpoint.safetensors").exists() {
        "bit_llama_checkpoint.safetensors".to_string()
    } else if Path::new("bit_llama/bit_llama_checkpoint.safetensors").exists() {
        println!("Using fallback checkpoint path: bit_llama/bit_llama_checkpoint.safetensors");
        "bit_llama/bit_llama_checkpoint.safetensors".to_string()
    } else {
        String::new()
    };

    if !checkpoint_path.is_empty() {
        println!("Resuming from checkpoint...");
        varmap.load(&checkpoint_path)?;
        let data = varmap.data().lock().expect("Failed to lock VarMap");
        println!("Checkpoint loaded. Key count: {}", data.len());
    } else {
        println!("No checkpoint found. Starting fresh.");
    }

    println!(
        "Model initialized. Varmap Key count: {}",
        varmap.data().lock().expect("Failed to lock VarMap").len()
    );

    let params = candle_nn::ParamsAdamW {
        lr: args.lr,
        ..Default::default()
    };
    let mut adam = candle_nn::AdamW::new(varmap.all_vars(), params)?;

    // 3. Step Persistence (Load) - with fallback path
    let state_path = if Path::new("training_state.json").exists() {
        "training_state.json".to_string()
    } else if Path::new("bit_llama/training_state.json").exists() {
        println!("Using fallback state path: bit_llama/training_state.json");
        "bit_llama/training_state.json".to_string()
    } else {
        "training_state.json".to_string() // Default for new runs
    };
    let mut start_step = 0;
    if Path::new(&state_path).exists() {
        if let Ok(file) = File::open(&state_path) {
            let reader = BufReader::new(file);
            if let Ok(json) = serde_json::from_reader::<_, serde_json::Value>(reader) {
                if let Some(s) = json.get("step").and_then(|v| v.as_u64()) {
                    start_step = s as usize;
                    println!("Resuming from Step {}", start_step);
                }
            }
        }
    }

    // 4. Loop
    let log_interval = 10;
    let save_interval = args.save_interval;
    println!(
        "Starting Training Loop (Target: {} steps, Save every {} steps)...",
        args.steps, save_interval
    );
    let total_steps = args.steps;

    // 【1】 実行制御用のフラグを作成 (スレッド間で共有するため Arc と AtomicBool を使う)
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    let ctrl_c_count = Arc::new(AtomicUsize::new(0));
    let c = ctrl_c_count.clone();

    // 【2】 Ctrl+C ハンドラをセット (2回押しで強制終了)
    ctrlc::set_handler(move || {
        let count = c.fetch_add(1, Ordering::SeqCst) + 1;
        if count == 1 {
            println!("\n\n🛑 Ctrl+C detected! Finishing current step and saving...");
            println!("   (Press Ctrl+C again to force quit WITHOUT saving)");
            r.store(false, Ordering::SeqCst);
        } else {
            println!("\n⚠️  Force quit! Exiting immediately without saving.");
            std::process::exit(1);
        }
    })
    .expect("Error setting Ctrl-C handler");

    // Best loss tracking
    let mut best_loss = f32::MAX;
    let mut checkpoint_history: Vec<String> = Vec::new(); // Rolling checkpoints

    for step in start_step..total_steps {
        // Automatic Warmup for Resume (100 steps)
        let current_lr = if start_step > 0 && step < start_step + 100 {
            let ratio = (step - start_step) as f64 / 100.0;
            args.lr * ratio
        } else {
            args.lr
        };
        adam.set_learning_rate(current_lr);

        if step < start_step + 5 || step % 10 == 0 {
            println!("Step {:4} | LR: {:.7} | Loading batch...", step, current_lr);
        }

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
        let _batch_loss_accum = 0.0;

        // Input: (B, T). We iterate t.
        for t in 0..CONTEXT_LEN {
            // Forward Pass
            let input_col = inputs.narrow(1, t, 1)?.squeeze(1)?.contiguous()?; // (B)
            let target_col = targets.narrow(1, t, 1)?.squeeze(1)?.contiguous()?; // (B)

            let mut h = model.embedding.forward(&input_col)?; // (B, D)

            // Pass through layers (TTT Update + Residual)
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

        // Log loss + Best model check
        let val = loss_scaled.to_scalar::<f32>()?;
        if step % log_interval == 0 {
            println!("Step {:4} | Loss: {:.4}", step, val);
        }

        // 🏆 Best Model Saving
        if val < best_loss {
            best_loss = val;
            println!(
                "🏆 New best loss: {:.4} - Saving model_best.safetensors",
                val
            );
            varmap.save("model_best.safetensors")?;
        }

        // ♻️ Save checkpoint at interval (Rolling Checkpoints)
        if step % save_interval == 0 && step > 0 {
            let checkpoint_name = format!("checkpoint_step_{}.safetensors", step);
            println!("[Saving checkpoint: {}...]", checkpoint_name);
            varmap.save(&checkpoint_name)?;
            let state = serde_json::json!({ "step": step });
            if let Ok(file) = File::create(&state_path) {
                serde_json::to_writer(file, &state)?;
            }

            // Rolling: keep only last 3 checkpoints
            checkpoint_history.push(checkpoint_name);
            if checkpoint_history.len() > 3 {
                let old = checkpoint_history.remove(0);
                if Path::new(&old).exists() {
                    println!("♻️ Deleting old checkpoint: {}", old);
                    let _ = std::fs::remove_file(&old);
                }
            }
        }

        // Check for Ctrl+C shutdown
        if !running.load(Ordering::SeqCst) {
            println!("[Shutdown] Saving checkpoint at step {}...", step);
            varmap.save("bit_llama_checkpoint.safetensors")?;
            let state = serde_json::json!({ "step": step });
            if let Ok(file) = File::create(&state_path) {
                serde_json::to_writer(file, &state)?;
            }
            println!("Exiting gracefully.");
            return Ok(());
        }
    }

    println!("Training complete. Saving final model...");
    varmap.save("bit_llama_v1.safetensors")?;

    Ok(())
}
