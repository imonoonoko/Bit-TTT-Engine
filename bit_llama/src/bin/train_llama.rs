use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use memmap2::Mmap; // Import memmap2
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

// Import core engine
// #[path = "../core_engine.rs"]
// mod core_engine;
use cortex_rust::{BitLlama, BitLlamaConfig};

// ============================================================
// Training Configuration Constants
// ============================================================
/// Maximum sequence length per batch (tokens).
/// Reduced for faster iteration; increase for better context modeling.
const CONTEXT_LEN: usize = 128;

/// Batch size. Optimized for 8GB VRAM GPUs.
const BATCH_SIZE: usize = 32;

/// Model hidden dimension (embedding size).
const DIM: usize = 256;

/// Number of transformer blocks (layers).
const LAYERS: usize = 4;

/// Vocabulary size. Must match the BPE tokenizer.
const VOCAB: usize = 16384;

// Data Loader
struct DataLoader {
    _file: File,     // Keep file handle alive (prefixed with _ to suppress warning)
    mmap: Mmap,      // Memory map
    data_len: usize, // count of u16 elements
    cursor: usize,
}

impl DataLoader {
    fn new(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // バイト数からu16の要素数を計算
        let data_len = mmap.len() / 2;

        Ok(Self {
            _file: file,
            mmap,
            data_len,
            cursor: 0,
        })
    }

    fn next_batch(
        &mut self,
        batch_size: usize,
        len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let mut inputs = Vec::with_capacity(batch_size * len);
        let mut targets = Vec::with_capacity(batch_size * len);

        for _ in 0..batch_size {
            if self.cursor + len + 1 >= self.data_len {
                self.cursor = 0; // Reset
            }

            // Mmapからスライスを取得し、u8 -> u16 -> u32 に変換
            let start = self.cursor * 2;
            let end = (self.cursor + len + 1) * 2;
            let chunk_u8 = &self.mmap[start..end];

            // chunk_u8 を u16 のイテレータとして扱う
            let chunk_u16: Vec<u16> = chunk_u8
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();

            // Fix: Candle doesn't support u16 tensors. Convert to u32.
            inputs.extend(chunk_u16[0..len].iter().map(|&x| x as u32));
            targets.extend(chunk_u16[1..len + 1].iter().map(|&x| x as u32));

            self.cursor += len;
        }

        let inp_tensor = Tensor::from_slice(&inputs, (batch_size, len), device)?;
        let tgt_tensor = Tensor::from_slice(&targets, (batch_size, len), device)?;

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

    /// 最小学習率 (Cosine Decayの着地点)
    #[arg(long, default_value_t = 0.0)]
    min_lr: f64,

    /// ウォームアップにかけるステップ数
    #[arg(long, default_value_t = 500)]
    warmup_steps: usize,

    #[arg(long)]
    load: Option<String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TrainingState {
    step: usize,
    loss: f32,
    #[allow(dead_code)]
    date: String,
    #[allow(dead_code)]
    checkpoint: String, // Relative path filename
}

fn save_training_state(
    base_dir: &str,
    filename_no_ext: &str,
    step: usize,
    loss: f32,
) -> Result<()> {
    let safetensors_name = format!("{}.safetensors", filename_no_ext);
    let json_name = format!("{}.json", filename_no_ext);

    let state = TrainingState {
        step,
        loss,
        date: chrono::Local::now().to_rfc3339(),
        checkpoint: safetensors_name,
    };

    let path = format!("{}{}", base_dir, json_name);
    let file = File::create(&path)?;
    serde_json::to_writer_pretty(file, &state)?;

    // Also update generic "training_state.json" for easy resuming
    let generic_path = format!("{}training_state.json", base_dir);
    if let Ok(file) = File::create(&generic_path) {
        let _ = serde_json::to_writer_pretty(file, &state);
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    // ============================================================
    // Section 1: Initialization
    // ============================================================
    println!("--- Bit-Llama Training (TinyStories) ---");
    println!(
        "Config: LR={}, Steps={}, Data={}",
        args.lr, args.steps, args.data
    );

    println!("Initializing Device...");
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Device initialized: {:?}", device);

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
    println!("Data Loaded. Total tokens: {}", loader.data_len);

    // ============================================================
    // Section 2: Model Setup
    // ============================================================
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
    // Determine base directory based on where we find the checkpoint
    let base_dir = if Path::new("bit_llama_checkpoint.safetensors").exists() {
        "".to_string()
    } else if Path::new("bit_llama/bit_llama_checkpoint.safetensors").exists() {
        println!("Using fallback directory: bit_llama/");
        "bit_llama/".to_string()
    } else if Path::new("bit_llama/training_state.json").exists() {
        // No checkpoint but state exists in bit_llama/
        println!("Using fallback directory: bit_llama/");
        "bit_llama/".to_string()
    } else {
        "".to_string() // New run, use current directory
    };

    let checkpoint_path = if let Some(path) = args.load {
        // A. Launcherで指定された場合 (最優先)
        println!("📂 Loading specific checkpoint from Launcher: {}", path);
        Some(path)
    } else if Path::new("bit_llama_checkpoint.safetensors").exists() {
        // B. カレントディレクトリにある場合
        Some("bit_llama_checkpoint.safetensors".to_string())
    } else if Path::new("bit_llama/bit_llama_checkpoint.safetensors").exists() {
        // C. サブフォルダにある場合
        Some("bit_llama/bit_llama_checkpoint.safetensors".to_string())
    } else {
        None
    };

    if let Some(path) = checkpoint_path {
        if Path::new(&path).exists() {
            println!("Resuming from checkpoint: {}", path);
            varmap.load(&path)?;
            // ... (ここから下の Key count 表示などはそのまま) ...
        } else {
            println!("⚠️ Specified checkpoint not found: {}", path);
        }
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

    // 3. Step Persistence (Load) - use same base_dir for consistency
    let state_path = format!("{}training_state.json", base_dir);
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

    // ============================================================
    // Section 3: Training Loop
    // ============================================================
    let log_interval = 10;
    let save_interval = args.save_interval;
    println!(
        "Starting Training Loop (Target: {} steps, Save every {} steps)...",
        args.steps, save_interval
    );

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

    // Verify CWD
    if let Ok(cwd) = std::env::current_dir() {
        println!("CWD: {:?}", cwd);
    }
    // --- Auto-Save Config ---
    let config_path = format!("{}config.json", base_dir);
    // Always overwrite to ensure it matches current run, or check if different?
    // Best practice: overwrite or ensure consistency. Let's just write it.
    let config_json = serde_json::json!({
        "dim": DIM,
        "hidden_dim": DIM,
        "n_layers": LAYERS,
        "n_heads": 8,
        "vocab_size": VOCAB,
        "norm_eps": 1e-5,
        "inner_lr": 0.01
    });
    if let Ok(f) = std::fs::File::create(&config_path) {
        let _ = serde_json::to_writer_pretty(f, &config_json);
        println!("💾 Saved training config to: {}", config_path);
    }

    // --- Auto-Save Tokenizer ---
    // Infer tokenizer path from data path (assuming it sits next to train.bin)
    let data_path_obj = Path::new(&args.data);
    let tokenizer_source = if let Some(parent) = data_path_obj.parent() {
        parent.join("tokenizer.json")
    } else {
        Path::new("data/TinyStories/tokenizer.json").to_path_buf()
    };

    let tokenizer_dest = format!("{}tokenizer.json", base_dir);
    if tokenizer_source.exists() {
        if let Err(e) = std::fs::copy(&tokenizer_source, &tokenizer_dest) {
            println!(
                "⚠️ Failed to copy tokenizer from {:?}: {}",
                tokenizer_source, e
            );
        } else {
            println!("✅ Tokenizer backed up to: {}", tokenizer_dest);
        }
    } else {
        // Fallback check for default location if inference failed
        let default_tok = Path::new("data/TinyStories/tokenizer.json");
        if default_tok.exists() && default_tok != tokenizer_source {
            if let Ok(_) = std::fs::copy(default_tok, &tokenizer_dest) {
                println!(
                    "✅ Tokenizer backed up to: {} (from default path)",
                    tokenizer_dest
                );
            }
        } else {
            println!(
                "⚠️ Warning: Source tokenizer not found at {:?}",
                tokenizer_source
            );
        }
    }

    // --- Training Loop ---
    for step in start_step..args.steps {
        // 0. Early Stop Check (Before Data Loading)
        if Path::new("stop_signal").exists() {
            println!("\n🛑 Stop signal detected (Start of Loop)! Saving and exiting...");
            let _ = std::fs::remove_file("stop_signal");
            varmap.save(&format!("{}bit_llama_checkpoint.safetensors", base_dir))?;
            let state = serde_json::json!({ "step": step });
            if let Ok(file) = File::create(&state_path) {
                serde_json::to_writer(file, &state)?;
            }
            return Ok(());
        }

        // Automatic Warmup for Resume (100 steps)
        // --- Learning Rate Schedule (Warmup + Cosine Decay) ---
        let current_lr = if step < args.warmup_steps {
            // 1. ウォームアップ期間 (0 -> Max LR)
            // 線形に加速します
            args.lr * (step as f64 / args.warmup_steps as f64)
        } else {
            // 2. 減衰期間 (Max LR -> Min LR)
            // 残りのステップ数に対する進捗率 (0.0 〜 1.0)
            let progress = (step - args.warmup_steps) as f64
                / (args.steps.saturating_sub(args.warmup_steps)) as f64;

            // 安全策: 1.0を超えないようにする
            let progress = progress.min(1.0).max(0.0);

            // Cosine計算: 1.0 (開始時) -> -1.0 (終了時)
            let cosine = (progress * std::f64::consts::PI).cos();

            // 減衰係数: 1.0 -> 0.0 に変換
            let decay = 0.5 * (1.0 + cosine);

            // 適用: 最小LR + (幅 * 係数)
            args.min_lr + (args.lr - args.min_lr) * decay
        };
        adam.set_learning_rate(current_lr);
        // -----------------------------------------------------

        /* Debug Logs - Temporarily commented out for speed
        if step < start_step + 5 || step % 10 == 0 {
            println!("Step {:4} | LR: {:.7} | Loading batch...", step, current_lr);
            std::io::stdout().flush().ok();
        }
        */

        let (inputs, targets) = loader.next_batch(BATCH_SIZE, CONTEXT_LEN, &device)?;
        /*
        if step < start_step + 5 || step % 10 == 0 {
            println!("Step {:4} | Batch loaded. Starting forward pass...", step);
            std::io::stdout().flush().ok();
        }
        */
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

        // let mut loss_step = Tensor::new(0.0f32, &device)?; // Unused initialization removed

        // Recursive Forward (Token by Token) -> NOW CHUNKWISE PARALLEL
        // let _batch_loss_accum = 0.0; // Unused

        let chunk_size = 32; // Mini-batch size for TTT

        // forward_chunkwise returns logits: (B, T, Vocab)
        let logits = model.forward_chunkwise(&inputs, &mut w_states, chunk_size)?;

        // Loss Calculation: Cross Entropy over Flattened Tensors
        // Logits: (B*T, V)
        // Targets: (B*T)
        let logits_flat = logits.reshape((BATCH_SIZE * CONTEXT_LEN, VOCAB))?;
        let targets_flat = targets.reshape(BATCH_SIZE * CONTEXT_LEN)?;

        // loss_step is currently (B*T) from cross_entropy?
        // Check candle docs: cross_entropy returns tensor.
        let loss_vec = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;
        let loss_step = loss_vec.mean_all()?; // Scalar average loss

        /*
        if step < start_step + 5 || step % 10 == 0 {
            println!(
                "Step {:4} | Forward complete. Starting backward pass...",
                step
            );
            std::io::stdout().flush().ok();
        }
        */

        // --- OPTIMIZATION: Check Loss only at log interval ---
        let mut loss_val_check = 0.0;

        // ログ出力タイミングのみ同期して値をチェックする (速度重視)
        if step % log_interval == 0 {
            loss_val_check = loss_step.to_scalar::<f32>()?;
            if loss_val_check.is_nan() {
                anyhow::bail!("Step {} | Loss is NaN! Stopping training.", step);
            }
        }
        // ------------------------------------------

        // Backprop on Mean Loss
        adam.backward_step(&loss_step)?;

        /*
        if step < start_step + 5 || step % 10 == 0 {
            println!("Step {:4} | Backward complete.", step);
            std::io::stdout().flush().ok();
        }
        */

        // Log loss
        if step % log_interval == 0 {
            let val = loss_val_check; // Already mean
            println!("Step {:4} | Loss: {:.4}", step, val);
            std::io::stdout().flush().ok();

            // 🏆 Best Model Saving (Check every 50 steps to reduce SSD writes)
            // Loss値をチェックした時だけ更新判定
            if step % 50 == 0 && step > 0 && val < best_loss {
                best_loss = val;
                println!(
                    "🏆 New best loss: {:.4} (Step {}) - Saving model_best.safetensors",
                    val, step
                );
                varmap.save(&format!("{}model_best.safetensors", base_dir))?;
                save_training_state(&base_dir, "model_best", step, val)?;
            }
        }

        // Check for 'stop_signal' file (Graceful Shutdown from GUI)
        let stop_path = Path::new("stop_signal");
        if stop_path.exists() {
            if let Ok(abs_path) = std::fs::canonicalize(stop_path) {
                println!(
                    "\n🛑 Stop signal detected at {:?}! Saving and exiting...",
                    abs_path
                );
            } else {
                println!("\n🛑 Stop signal detected! Saving and exiting...");
            }

            // 1. Remove signal file
            let _ = std::fs::remove_file("stop_signal");

            // 2. Save checkpoint
            // 2. Save checkpoint
            varmap.save(&format!("{}bit_llama_checkpoint.safetensors", base_dir))?;
            save_training_state(&base_dir, "bit_llama_checkpoint", step, loss_val_check)?;

            println!("✅ Saved successfully. Exiting.");
            return Ok(());
        }

        // ♻️ Save checkpoint at interval (Rolling Checkpoints)
        if step % save_interval == 0 && step > 0 {
            let filename_no_ext = format!("{}checkpoint_step_{}", base_dir, step);
            let safetensors_path = format!("{}.safetensors", filename_no_ext);

            println!("[Saving checkpoint: {}...]", safetensors_path);
            varmap.save(&safetensors_path)?;

            // Helper updates both specific JSON and generic training_state.json
            save_training_state(
                &base_dir,
                &format!("checkpoint_step_{}", step),
                step,
                loss_val_check,
            )?;

            // Rolling: keep only last 3 checkpoints
            checkpoint_history.push(safetensors_path);
            if checkpoint_history.len() > 3 {
                let old = checkpoint_history.remove(0);
                if Path::new(&old).exists() {
                    let _ = std::fs::remove_file(&old);
                    // Also remove corresponding json
                    let old_json = old.replace(".safetensors", ".json");
                    if Path::new(&old_json).exists() {
                        let _ = std::fs::remove_file(&old_json);
                    }
                }
            }
        }

        // Check for Ctrl+C shutdown
        if !running.load(Ordering::SeqCst) {
            println!("[Shutdown] Saving checkpoint at step {}...", step);
            varmap.save(&format!("{}bit_llama_checkpoint.safetensors", base_dir))?;
            let state = serde_json::json!({ "step": step });
            if let Ok(file) = File::create(&state_path) {
                serde_json::to_writer(file, &state)?;
            }
            println!("Exiting gracefully.");
            return Ok(());
        }
    }

    println!("Training complete. Saving final model...");
    varmap.save(&format!("{}bit_llama_v1.safetensors", base_dir))?;

    Ok(())
}
