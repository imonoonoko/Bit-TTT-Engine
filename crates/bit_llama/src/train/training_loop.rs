//! Training Loop - Main training execution (MeZO Implementation)

use anyhow::Result;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{ops, VarBuilder, VarMap};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use cortex_rust::BitLlama;
use tokenizers::Tokenizer;
use tracing::{error, info, warn};

use super::args::TrainArgs;
use super::checkpoint::{find_checkpoint_path, load_start_step, save_training_state};
use crate::loader::BitLoader;
use fs2::FileExt;

fn save_securely(varmap: &VarMap, path: &str) -> Result<()> {
    let lock_path = format!("{path}.lock");
    let lock_file = File::create(&lock_path)?;
    lock_file.lock_exclusive()?;
    varmap.save(path)?;
    lock_file.unlock()?;
    Ok(())
}

/// `MeZO`: Perturb weights using a deterministic seed.
/// vars: List of model variables
/// seed: Random seed (u64)
/// scale: scaling factor (epsilon or -lr * grad)
/// If scale is 0, does nothing.
///
/// # Panics
/// This function panics if the normal distribution cannot be created (e.g. invalid parameters), though parameters are hardcoded to standard normal.
fn perturb_weights(vars: &[Var], seed: u64, scale: f64) -> Result<()> {
    if scale == 0.0 {
        return Ok(());
    }

    // We iterate over all variables.
    // To ensure determinism, we seed the RNG for EACH variable uniquely based on the global seed + var index.
    // This allows us to parallelize if needed (though current impl is serial loop) and ensures consistency.
    // Actually, creating a new RNG for each var is expensive.
    // Better: Creating one RNG seeded with `seed` and pulling from it sequentially.
    // REQUIRED: `vars` order must be deterministic. `VarMap::all_vars()` returns vars in insertion order (usually).
    // Given the model structure is static, this should be stable.

    let mut rng = StdRng::seed_from_u64(seed);
    // Standard Normal Distribution
    let normal = Normal::new(0.0, 1.0).unwrap();

    for var in vars {
        let shape = var.shape();
        let _dims = shape.dims();
        let elem_count = shape.elem_count();

        // Generate noise on CPU then move to Device? Or use Candle's random if possible.
        // Candle `Tensor::randn` uses the device's generator if available, or CPU.
        // MeZO paper suggests: Z ~ N(0, 1).
        // To be memory efficient, we don't store Z. We generate it on the fly.
        // `perturb_weights` is called 3 times per step with SAME seed.
        // So we must generate exact same Z sequence.

        // Strategy: Use `rand` crate to generate a seed for Candle's random?
        // Candle `randn` takes a seed? `candle_core::utils::set_seed` is global.
        // Using global seed in a loop is risky if logic changes.

        // Fallback: Generate generic noise using `rand_distr` into a Vec<f32>, convert to Tensor, add.
        // This allocates O(N) memory for noise.
        // MeZO benefit is O(1) memory *stored* (activations).
        // Having a temporary O(N) buffer for *one* layer's noise is fine.
        // We do it layer by layer (var by var).

        // Optimization: Pre-allocate a buffer?
        // For now, simple vector generation.

        let noise_vec: Vec<f32> = (0..elem_count)
            .map(|_| {
                #[allow(clippy::cast_possible_truncation)]
                let sample = normal.sample(&mut rng) as f32;
                sample
            })
            .collect();

        let noise_tensor = Tensor::from_vec(noise_vec, shape, var.device())?;

        // Update: theta = theta + scale * Z
        // var = var + (scale * noise)
        let scaled_noise = (noise_tensor * scale)?;
        let new_val = (var.as_tensor() + scaled_noise)?;
        var.set(&new_val)?;
    }

    Ok(())
}

/// Main training function
pub fn run(args: TrainArgs) -> Result<()> {
    // ============================================================
    // Section 1: Initialization
    // ============================================================
    info!("--- Bit-Llama Training (MeZO - Memory Efficient) ---");
    info!(
        "Config: Dim={}, Layers={}, Context={}, Batch={}",
        args.dim, args.layers, args.context_len, args.batch_size
    );
    info!(
        "Hyperparams: LR={}, Steps={}, Warmup={}, Epsilon={}",
        args.lr, args.steps, args.warmup_steps, args.epsilon
    );

    let params_est = 12 * args.layers * args.dim * args.dim;
    let params_m = params_est as f64 / 1_000_000.0;

    let vram_est_mb = (params_est as f64 * 4.0) / (1024.0 * 1024.0); // F32 weights
                                                                     // MeZO Act memory is negligible (inference mode), mostly KV cache or temp buffers.
                                                                     // Say 2 * Context * Layers * Dim for KV?
                                                                     // BitLinear uses some buffer.
                                                                     // Estimate: Weights + 512MB Buffer.
    let total_vram_est = vram_est_mb + 512.0;

    info!("📊 Model Size: {:.2}M Params", params_m);
    info!(
        "💾 Est. VRAM Usage: {:.2} MB (MeZO Enabled)",
        total_vram_est
    );

    info!("Initializing Device...");
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    info!("Device initialized: {:?}", device);

    let mut data_path = args.data.clone();
    if !Path::new(&data_path).exists() {
        let fallback = format!("bit_llama/{}", &args.data);
        if Path::new(&fallback).exists() {
            data_path = fallback;
            info!("Using fallback data path: {}", data_path);
        } else {
            anyhow::bail!("Data not found at '{}' or '{}'", args.data, fallback);
        }
    }

    let data_path_obj = Path::new(&data_path);
    let tokenizer_path = if data_path_obj.is_dir() {
        data_path_obj.join("tokenizer.json")
    } else if let Some(parent) = data_path_obj.parent() {
        parent.join("tokenizer.json")
    } else {
        Path::new("workspace/data/TinyStories/tokenizer.json").to_path_buf()
    };

    info!("Loading Tokenizer from: {:?}", tokenizer_path);
    let vocab_size = if tokenizer_path.exists() {
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let v = tokenizer.get_vocab_size(true);
        info!("✅ Tokenizer Loaded. Vocab Size: {}", v);
        v
    } else {
        warn!(
            "⚠️ Tokenizer not found! Specific path: {}",
            tokenizer_path.display()
        );
        warn!("⚠️ Defaulting VOCAB to 16384 (Risk of mismatch!)");
        16384
    };

    let loader_path = if data_path_obj.is_dir() {
        let p_u32 = data_path_obj.join("train.u32");
        if p_u32.exists() {
            p_u32.to_string_lossy().to_string()
        } else {
            let p_txt = data_path_obj.join("train.txt");
            if p_txt.exists() {
                p_txt.to_string_lossy().to_string()
            } else {
                let p_tiny = data_path_obj.join("tiny_stories_train.txt");
                if p_tiny.exists() {
                    p_tiny.to_string_lossy().to_string()
                } else {
                    anyhow::bail!(
                        "No train.u32, train.txt, or tiny_stories_train.txt found in {:?}",
                        data_path
                    );
                }
            }
        }
    } else {
        data_path.clone()
    };

    let mut loader = BitLoader::new(&loader_path)?;
    info!("Data Loaded. Total tokens: {}", loader.data_len);
    if let Some(_) = loader.mask_mmap {
        info!("✅ Mask file detected and loaded.");
    }

    if loader.data_len == 0 {
        anyhow::bail!("❌ Training dataset is empty! Please check your input files and run Preprocessing again.");
    }

    let mut project_config = crate::config::ProjectConfig::from_args(&args);
    // Override fields not present in TrainArgs
    project_config.vocab_size = vocab_size;

    let config = project_config.to_bit_llama_config(0.1);

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = BitLlama::load(config.clone(), vb)?;

    let base_dir = if Path::new("bit_llama_checkpoint.safetensors").exists() {
        "".to_string()
    } else if Path::new("bit_llama/bit_llama_checkpoint.safetensors").exists() {
        info!("Using fallback directory: bit_llama/");
        "bit_llama/".to_string()
    } else if Path::new("bit_llama/training_state.json").exists() {
        info!("Using fallback directory: bit_llama/");
        "bit_llama/".to_string()
    } else {
        "".to_string()
    };

    let checkpoint_path = find_checkpoint_path(args.load.as_ref(), &base_dir);

    if let Some(path) = checkpoint_path {
        if Path::new(&path).exists() {
            info!("Resuming from checkpoint: {}", path);
            match varmap.load(&path) {
                Ok(_) => {
                    info!("✅ Checkpoint loaded successfully.");
                }
                Err(e) => {
                    if args.load.is_some() {
                        // If user explicitly requested this checkpoint, fail hard.
                        anyhow::bail!("❌ Failed to load requested checkpoint '{}': {}", path, e);
                    } else {
                        // Auto-resume failed (likely shape mismatch or corrupt). Start fresh.
                        warn!("⚠️ Failed to load auto-checkpoint '{}': {}", path, e);
                        warn!("⚠️ Likely shape mismatch or corrupt file. Starting fresh instead.");
                    }
                }
            }
        } else {
            warn!("⚠️ Specified checkpoint not found: {}", path);
        }
    } else {
        info!("No checkpoint found. Starting fresh.");
    }

    info!(
        "Model initialized. Varmap Key count: {}",
        varmap.data().lock().expect("Failed to lock VarMap").len()
    );

    let optim_vars = varmap.all_vars();

    // RNG for MeZO noise (Step Seed)
    let mut step_rng = StdRng::from_entropy();

    let start_step = load_start_step(&base_dir);
    if start_step > 0 {
        info!("Resuming from Step {}", start_step);
    }

    let log_interval = 10;
    let save_interval = args.save_interval;

    info!(
        "Starting MeZO Training Loop (Target: {} steps, Epsilon: {})",
        args.steps, args.epsilon
    );

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    let ctrl_c_count = Arc::new(AtomicUsize::new(0));
    let c = ctrl_c_count.clone();

    ctrlc::set_handler(move || {
        let count = c.fetch_add(1, Ordering::SeqCst) + 1;
        if count == 1 {
            info!("\n\n🛑 Ctrl+C detected! Finishing current step and saving...");
            info!("   (Press Ctrl+C again to force quit WITHOUT saving)");
            r.store(false, Ordering::SeqCst);
        } else {
            error!("\n⚠️  Force quit! Exiting immediately without saving.");
            std::process::exit(1);
        }
    })
    .expect("Error setting Ctrl-C handler");

    let mut best_loss = f32::MAX;
    let mut checkpoint_history: Vec<String> = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        info!("CWD: {:?}", cwd);
    }

    let effective_output_dir = if let Some(ref output_dir) = args.output_dir {
        let dir = if output_dir.ends_with('/') || output_dir.ends_with('\\') {
            output_dir.clone()
        } else {
            format!("{}/", output_dir)
        };
        if let Err(e) = std::fs::create_dir_all(output_dir) {
            warn!("⚠️ Failed to create output directory: {}", e);
        } else {
            info!("📁 Output directory: {}", output_dir);
        }
        dir
    } else {
        base_dir.clone()
    };

    let start_time = std::time::Instant::now();
    let state_path = format!("{}training_state.json", base_dir);
    let epsilon = args.epsilon;

    // Mock Mode Setup

    for step in start_step..args.steps {
        // Mock Mode Loop
        if args.mock {
            std::thread::sleep(std::time::Duration::from_millis(100));
            // Generate dummy loss: Curve down + Noise
            let progress = step as f32 / args.steps as f32;
            let mock_loss = 2.5 - (2.0 * progress) + (step_rng.gen::<f32>() * 0.2);

            if step % log_interval == 0 {
                info!(
                    "Step {:4} | Loss: {:.4} | LR: {:.7} | MeZO Grad: 0.00e+00 | 0.00 tok/s",
                    step, mock_loss, 0.0001
                );
                // Report VRAM for Mock
                info!("       [VRAM] Used: 123.45 MB (Mock)");
            }
            // Handle Stop Signal in Mock
            if Path::new("stop_signal").exists() {
                info!("\n🛑 Stop signal detected (Mock)! Exiting...");
                let _ = std::fs::remove_file("stop_signal");
                return Ok(());
            }
            continue;
        }

        if Path::new("stop_signal").exists() {
            info!("\n🛑 Stop signal detected (Start of Loop)! Saving and exiting...");
            let _ = std::fs::remove_file("stop_signal");
            save_securely(
                &varmap,
                &format!("{}bit_llama_checkpoint.safetensors", base_dir),
            )?;
            let state = serde_json::json!({ "step": step });
            if let Ok(file) = File::create(&state_path) {
                serde_json::to_writer(file, &state)?;
            }
            return Ok(());
        }

        // Update LR
        let current_lr = if step < args.warmup_steps {
            args.lr * (step as f64 / args.warmup_steps as f64)
        } else {
            let progress = (step - args.warmup_steps) as f64
                / (args.steps.saturating_sub(args.warmup_steps)) as f64;
            let progress = progress.clamp(0.0, 1.0);
            let cosine = (progress * std::f64::consts::PI).cos();
            let decay = 0.5 * (1.0 + cosine);
            args.min_lr + (args.lr - args.min_lr) * decay
        };

        // Load Batch
        let (inputs, targets, mask_tensor) =
            loader.next_batch_masked(args.batch_size, args.context_len, &device)?;

        // ====================================================================
        // MeZO Protocol
        // ====================================================================
        let seed = step_rng.gen::<u64>();

        // 1. Perturb (+)
        // theta = theta + epsilon * Z
        perturb_weights(&optim_vars, seed, epsilon)?;

        // Forward (+ loop)
        let loss_pos = {
            let d_small = args.dim / 4;
            let mut w_states = Vec::new(); // Reset states
            for _ in 0..args.layers {
                w_states.push(Tensor::zeros(
                    (args.batch_size, d_small, d_small),
                    DType::F32,
                    &device,
                )?);
            }
            let chunk_size = 32;
            let logits = model.forward_chunkwise(&inputs, &mut w_states, chunk_size)?;
            let logits_flat =
                logits.reshape((args.batch_size * args.context_len, config.vocab_size))?;
            let targets_flat = targets.reshape(args.batch_size * args.context_len)?;

            // Manual cross_entropy to ensure element-wise loss (for masking)
            let log_sm = ops::log_softmax(&logits_flat, candle_core::D::Minus1)?;
            let loss_vec = log_sm
                .gather(&targets_flat.unsqueeze(1)?, candle_core::D::Minus1)?
                .squeeze(candle_core::D::Minus1)?
                .neg()?;

            if let Some(ref m) = mask_tensor {
                let m_flat = m.reshape(loss_vec.shape())?;
                let masked_loss = (loss_vec * m_flat.clone())?;
                let sum_loss = masked_loss.sum_all()?.to_scalar::<f32>()?;
                let sum_mask = m_flat.sum_all()?.to_scalar::<f32>()?;
                if sum_mask == 0.0 {
                    0.0
                } else {
                    sum_loss / sum_mask
                }
            } else {
                loss_vec.mean_all()?.to_scalar::<f32>()?
            }
        };

        // 2. Perturb (-)
        // theta = (theta + epsilon * Z) - 2 * epsilon * Z = theta - epsilon * Z
        perturb_weights(&optim_vars, seed, -2.0 * epsilon)?;

        // Forward (- loop)
        let loss_neg = {
            let d_small = args.dim / 4;
            let mut w_states = Vec::new(); // Reset states (Independent forward)
            for _ in 0..args.layers {
                w_states.push(Tensor::zeros(
                    (args.batch_size, d_small, d_small),
                    DType::F32,
                    &device,
                )?);
            }
            let chunk_size = 32;
            let logits = model.forward_chunkwise(&inputs, &mut w_states, chunk_size)?;
            let logits_flat =
                logits.reshape((args.batch_size * args.context_len, config.vocab_size))?;
            let targets_flat = targets.reshape(args.batch_size * args.context_len)?;

            // Manual cross_entropy to ensure element-wise loss (for masking)
            let log_sm = ops::log_softmax(&logits_flat, candle_core::D::Minus1)?;
            let loss_vec = log_sm
                .gather(&targets_flat.unsqueeze(1)?, candle_core::D::Minus1)?
                .squeeze(candle_core::D::Minus1)?
                .neg()?;

            if let Some(ref m) = mask_tensor {
                let m_flat = m.reshape(loss_vec.shape())?;
                let masked_loss = (loss_vec * m_flat.clone())?;
                let sum_loss = masked_loss.sum_all()?.to_scalar::<f32>()?;
                let sum_mask = m_flat.sum_all()?.to_scalar::<f32>()?;
                if sum_mask == 0.0 {
                    0.0
                } else {
                    sum_loss / sum_mask
                }
            } else {
                loss_vec.mean_all()?.to_scalar::<f32>()?
            }
        };

        // 3. Restore
        // theta = (theta - epsilon * Z) + epsilon * Z = theta
        perturb_weights(&optim_vars, seed, epsilon)?;

        // 4. Update
        // projected_grad = (loss_pos - loss_neg) / (2 * epsilon)
        // theta = theta - lr * projected_grad * Z
        // Can be written as: perturb(seed, -lr * projected_grad)

        let projected_grad = (loss_pos - loss_neg) / (2.0 * epsilon as f32);
        let update_scale = -current_lr * projected_grad as f64;

        perturb_weights(&optim_vars, seed, update_scale)?;

        // ====================================================================

        if step % log_interval == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let avg_tokens_per_sec = if elapsed > 0.0 {
                ((step - start_step + 1) as f64 / elapsed)
                    * args.batch_size as f64
                    * args.context_len as f64
            } else {
                0.0
            };

            // Using loss_pos as proxy for current loss, though it's perturbed.
            // Or avg of pos/neg? Or separate forward?
            // Separate forward is expensive. Use loss_pos.

            info!(
                "Step {:4} | Loss: {:.4} | LR: {:.7} | MeZO Grad: {:.2e} | {:.2} tok/s",
                step, loss_pos, current_lr, projected_grad, avg_tokens_per_sec
            );

            // Debug VRAM
            if let Ok((free, total)) = cortex_rust::device_utils::get_vram_info(0) {
                let used_mb = (total - free) as f64 / 1024.0 / 1024.0;
                info!("       [VRAM] Used: {:.2} MB (Should be stable)", used_mb);
            }

            // Checkpoint Logic: Best Model
            if step > 0 && loss_pos < best_loss {
                best_loss = loss_pos;
                info!("🌟 New Best Loss: {:.4}", best_loss);
                save_securely(
                    &varmap,
                    &format!("{}model-best.safetensors", effective_output_dir),
                )?;
            }
        }

        // ... (Cleanup: Existing Save Logic check) ...
        // Keeping it minimal for MeZO refactor to fit in replacement limit.
        // I must reimplement the save logic or it gets deleted.

        if !args.benchmark && step % save_interval == 0 && step > 0 {
            let filename_no_ext = format!("{}checkpoint_step_{}", effective_output_dir, step);
            let safetensors_path = format!("{}.safetensors", filename_no_ext);

            save_securely(&varmap, &safetensors_path)?;
            // Also save as "latest"
            save_securely(
                &varmap,
                &format!("{}model-latest.safetensors", effective_output_dir),
            )?;

            save_training_state(
                &effective_output_dir,
                &format!("checkpoint_step_{}", step),
                step,
                loss_pos,
            )?;

            // Rotate
            checkpoint_history.push(safetensors_path);
            if checkpoint_history.len() > 3 {
                let old = checkpoint_history.remove(0);
                if Path::new(&old).exists() {
                    let _ = std::fs::remove_file(&old);
                }
            }
        }

        if !running.load(Ordering::SeqCst) {
            info!("[Shutdown] Saving checkpoint at step {}...", step);
            save_securely(
                &varmap,
                &format!("{}bit_llama_checkpoint.safetensors", base_dir),
            )?;
            let state = serde_json::json!({ "step": step });
            if let Ok(file) = File::create(&state_path) {
                serde_json::to_writer(file, &state)?;
            }
            info!("Exiting gracefully.");
            return Ok(());
        }
    }

    info!("Training complete. Saving final model...");
    // Final save logic...
    if let Some(ref output_dir) = args.output_dir {
        let model_path = format!("{}/model.safetensors", output_dir);
        save_securely(&varmap, &model_path)?;
    } else {
        save_securely(&varmap, &format!("{}bit_llama_v1.safetensors", base_dir))?;
    }

    Ok(())
}
