//! Training Loop - Main training execution

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
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

/// Main training function
pub fn run(args: TrainArgs) -> Result<()> {
    // ============================================================
    // Section 1: Initialization
    // ============================================================
    info!("--- Bit-Llama Training (Pure Rust) ---");
    info!(
        "Config: Dim={}, Layers={}, Context={}, Batch={}",
        args.dim, args.layers, args.context_len, args.batch_size
    );
    info!(
        "Hyperparams: LR={}, Steps={}, Warmup={}, MinLR={}",
        args.lr, args.steps, args.warmup_steps, args.min_lr
    );

    let params_est = 12 * args.layers * args.dim * args.dim;
    let params_m = params_est as f64 / 1_000_000.0;

    let vram_est_mb = (params_est as f64 * 16.0) / (1024.0 * 1024.0);
    let act_est_mb = (args.context_len * args.batch_size * args.dim * args.layers * 4) as f64
        / (1024.0 * 1024.0);
    let total_vram_est = vram_est_mb + act_est_mb + 512.0;

    info!("📊 Model Size: {:.2}M Params", params_m);
    info!(
        "💾 Est. VRAM Usage: {:.2} MB (Params: {:.2} + Act: {:.2})",
        total_vram_est, vram_est_mb, act_est_mb
    );

    if total_vram_est > 8000.0 {
        warn!("⚠️  WARNING: VRAM usage might exceed 8GB GPU limits!");
    }

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
        Path::new("data/TinyStories/tokenizer.json").to_path_buf()
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
            "⚠️ Tokenizer not found! Specific path: {:?}",
            tokenizer_path
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

    let project_config = crate::config::ProjectConfig {
        name: "Training Run".to_string(),
        created_at: chrono::Local::now().to_string(),
        vocab_size,
        val_ratio: 0.05,
        model_dim: args.dim,
        layers: args.layers,
        context_len: args.context_len,
        n_heads: 8,
        batch_size: args.batch_size,
        steps: args.steps,
        lr: args.lr,
        min_lr: args.min_lr,
        warmup_steps: args.warmup_steps,
        save_interval: args.save_interval,
    };

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
            varmap.load(&path)?;
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

    let params = candle_nn::ParamsAdamW {
        lr: args.lr,
        ..Default::default()
    };
    let mut adam = candle_nn::AdamW::new(varmap.all_vars(), params)?;

    let start_step = load_start_step(&base_dir);
    if start_step > 0 {
        info!("Resuming from Step {}", start_step);
    }

    let log_interval = 10;
    let save_interval = args.save_interval;
    info!(
        "Starting Training Loop (Target: {} steps, Save every {} steps)...",
        args.steps, save_interval
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

    if args.benchmark {
        info!("\n⚡ BENCHMARK MODE ENABLED ⚡");
        info!("   - Checkpoints: DISABLED");
        info!("   - State Saving: DISABLED");
        info!("   - Target: {} steps", args.steps);
    } else {
        let project_config_json = serde_json::to_string_pretty(&project_config)?;
        let project_config_path = format!("{}project.json", effective_output_dir);
        std::fs::write(&project_config_path, project_config_json)?;
        info!("✅ Project Config saved to: {}", project_config_path);

        let config_json = serde_json::to_string_pretty(&config)?;
        let config_path = format!("{}config.json", effective_output_dir);
        std::fs::write(&config_path, config_json)?;
        info!("✅ Model Config saved to: {}", config_path);

        let data_path_obj = Path::new(&args.data);
        let tokenizer_source = if data_path_obj.is_dir() {
            data_path_obj.join("tokenizer.json")
        } else if let Some(parent) = data_path_obj.parent() {
            parent.join("tokenizer.json")
        } else {
            Path::new("data/TinyStories/tokenizer.json").to_path_buf()
        };

        let tokenizer_dest = format!("{}tokenizer.json", effective_output_dir);
        if tokenizer_source.exists() {
            if let Err(e) = std::fs::copy(&tokenizer_source, &tokenizer_dest) {
                warn!(
                    "⚠️ Failed to copy tokenizer from {:?}: {}",
                    tokenizer_source, e
                );
            } else {
                info!("✅ Tokenizer backed up to: {}", tokenizer_dest);
            }
        } else {
            let default_tok = Path::new("data/TinyStories/tokenizer.json");
            if default_tok.exists() && default_tok != tokenizer_source {
                if std::fs::copy(default_tok, &tokenizer_dest).is_ok() {
                    info!(
                        "✅ Tokenizer backed up to: {} (from default path)",
                        tokenizer_dest
                    );
                }
            } else {
                warn!(
                    "⚠️ Warning: Source tokenizer not found at {:?}",
                    tokenizer_source
                );
            }
        }
    }

    let start_time = std::time::Instant::now();
    let state_path = format!("{}training_state.json", base_dir);

    for step in start_step..args.steps {
        if Path::new("stop_signal").exists() {
            info!("\n🛑 Stop signal detected (Start of Loop)! Saving and exiting...");
            let _ = std::fs::remove_file("stop_signal");
            varmap.save(&format!("{}bit_llama_checkpoint.safetensors", base_dir))?;
            let state = serde_json::json!({ "step": step });
            if let Ok(file) = File::create(&state_path) {
                serde_json::to_writer(file, &state)?;
            }
            return Ok(());
        }

        let current_lr = if step < args.warmup_steps {
            args.lr * (step as f64 / args.warmup_steps as f64)
        } else {
            let progress = (step - args.warmup_steps) as f64
                / (args.steps.saturating_sub(args.warmup_steps)) as f64;
            let progress = progress.min(1.0).max(0.0);
            let cosine = (progress * std::f64::consts::PI).cos();
            let decay = 0.5 * (1.0 + cosine);
            args.min_lr + (args.lr - args.min_lr) * decay
        };
        adam.set_learning_rate(current_lr);

        let (inputs, targets) = loader.next_batch(args.batch_size, args.context_len, &device)?;

        let d_small = args.dim / 4;
        let mut w_states = Vec::new();
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
        let loss_vec = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;
        let loss_step = loss_vec.mean_all()?;

        let mut loss_val_check = 0.0;
        if step % log_interval == 0 {
            loss_val_check = loss_step.to_scalar::<f32>()?;
            if loss_val_check.is_nan() {
                anyhow::bail!("Step {} | Loss is NaN! Stopping training.", step);
            }
        }

        adam.backward_step(&loss_step)?;

        if step % log_interval == 0 {
            let val = loss_val_check;
            let elapsed = start_time.elapsed().as_secs_f64();
            let avg_tokens_per_sec = if elapsed > 0.0 {
                ((step - start_step + 1) as f64 / elapsed)
                    * args.batch_size as f64
                    * args.context_len as f64
            } else {
                0.0
            };

            info!(
                "Step {:4} | Loss: {:.4} | LR: {:.7} | {:.2} tok/s",
                step, val, current_lr, avg_tokens_per_sec
            );

            if args.benchmark {
                continue;
            }

            if step % 50 == 0 && step > 0 && val < best_loss {
                best_loss = val;
                info!(
                    "🏆 New best loss: {:.4} (Step {}) - Saving model_best.safetensors",
                    val, step
                );
                varmap.save(&format!("{}model_best.safetensors", effective_output_dir))?;
                save_training_state(&effective_output_dir, "model_best", step, val)?;
            }
        }

        let stop_path = Path::new("stop_signal");
        if stop_path.exists() {
            info!("\n🛑 Stop signal detected! Saving and exiting...");
            let _ = std::fs::remove_file("stop_signal");
            varmap.save(&format!(
                "{}bit_llama_checkpoint.safetensors",
                effective_output_dir
            ))?;
            save_training_state(
                &effective_output_dir,
                "bit_llama_checkpoint",
                step,
                loss_val_check,
            )?;
            info!("✅ Saved successfully. Exiting.");
            return Ok(());
        }

        if !args.benchmark && step % save_interval == 0 && step > 0 {
            let filename_no_ext = format!("{}checkpoint_step_{}", effective_output_dir, step);
            let safetensors_path = format!("{}.safetensors", filename_no_ext);

            info!("[Saving checkpoint: {}...]", safetensors_path);
            varmap.save(&safetensors_path)?;
            save_training_state(
                &effective_output_dir,
                &format!("checkpoint_step_{}", step),
                step,
                loss_val_check,
            )?;

            checkpoint_history.push(safetensors_path);
            if checkpoint_history.len() > 3 {
                let old = checkpoint_history.remove(0);
                if Path::new(&old).exists() {
                    let _ = std::fs::remove_file(&old);
                    let old_json = old.replace(".safetensors", ".json");
                    if Path::new(&old_json).exists() {
                        let _ = std::fs::remove_file(&old_json);
                    }
                }
            }
        }

        if !running.load(Ordering::SeqCst) {
            info!("[Shutdown] Saving checkpoint at step {}...", step);
            varmap.save(&format!("{}bit_llama_checkpoint.safetensors", base_dir))?;
            let state = serde_json::json!({ "step": step });
            if let Ok(file) = File::create(&state_path) {
                serde_json::to_writer(file, &state)?;
            }
            info!("Exiting gracefully.");
            return Ok(());
        }
    }

    info!("Training complete. Saving final model...");

    if let Some(ref output_dir) = args.output_dir {
        let dir = if output_dir.ends_with('/') || output_dir.ends_with('\\') {
            output_dir.clone()
        } else {
            format!("{}/", output_dir)
        };
        std::fs::create_dir_all(output_dir)?;

        let model_path = format!("{}model.safetensors", dir);
        varmap.save(&model_path)?;
        info!("✅ Model saved to: {}", model_path);

        let src_tokenizer = data_path_obj.join("tokenizer.json");
        let dst_tokenizer = format!("{}tokenizer.json", dir);
        if src_tokenizer.exists() {
            std::fs::copy(&src_tokenizer, &dst_tokenizer)?;
            info!("✅ Tokenizer copied to: {}", dst_tokenizer);
        }

        let project_config_json = serde_json::to_string_pretty(&project_config)?;
        let project_config_path = format!("{}project.json", dir);
        std::fs::write(&project_config_path, project_config_json)?;
        info!("✅ Project Config saved to: {}", project_config_path);

        let config_json = serde_json::to_string_pretty(&config)?;
        let config_path = format!("{}config.json", dir);
        std::fs::write(&config_path, config_json)?;
        info!("✅ Model Config saved to: {}", config_path);
    } else {
        varmap.save(&format!("{}bit_llama_v1.safetensors", base_dir))?;
    }

    Ok(())
}
