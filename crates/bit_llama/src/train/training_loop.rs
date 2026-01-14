//! Training Loop - Main training execution

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
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
    let lock_path = format!("{}.lock", path);
    let lock_file = File::create(&lock_path)?;
    lock_file.lock_exclusive()?;
    varmap.save(path)?;
    lock_file.unlock()?;
    Ok(())
}

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

    let params_sf = cortex_rust::optim::schedule_free::ParamsScheduleFree {
        lr: args.lr,
        warmup_steps: args.warmup_steps,
        ..Default::default()
    };
    let optim_vars = varmap.all_vars();
    let mut optimizer = cortex_rust::optim::schedule_free::ScheduleFreeOptimizer::new(
        optim_vars.clone(),
        params_sf,
    )?;

    let start_step = load_start_step(&base_dir);
    if start_step > 0 {
        info!("Resuming from Step {}", start_step);
    }

    let log_interval = 10;
    let save_interval = args.save_interval;
    let accumulation_steps = args.accum.max(1);

    info!(
        "Starting Training Loop (Target: {} steps, Accum: {})",
        args.steps, accumulation_steps
    );

    // Gradient Accumulator Buffer (Matches optim_vars order)
    let mut grad_accumulator: Vec<Option<Tensor>> = vec![None; optim_vars.len()];

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
            let progress = progress.min(1.0).max(0.0);
            let cosine = (progress * std::f64::consts::PI).cos();
            let decay = 0.5 * (1.0 + cosine);
            args.min_lr + (args.lr - args.min_lr) * decay
        };
        optimizer.set_learning_rate(current_lr);

        // Schedule-Free: Pre-step (Interpolate y)
        // Only needed at start of accumulation cycle
        if step % accumulation_steps == 0 {
            optimizer.pre_step()?;
        }

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

        // Backward & Accumulate
        let grads_store = loss_step.backward()?;
        for (i, var) in optim_vars.iter().enumerate() {
            if let Some(grad) = grads_store.get(var) {
                // Normalize by accumulation steps to keep scale correct
                // BUT usually we just sum and divide LR?
                // Standard: loss / accum.
                // Here we didn't divide loss. So gradients are huge.
                // We should divide accumulated gradients by accum_steps in Optimizer?
                // Or here?
                // Let's divide here.
                let scaled_grad = (grad / (accumulation_steps as f64))?;
                match &grad_accumulator[i] {
                    Some(acc) => {
                        let new_acc = (acc + &scaled_grad)?;
                        grad_accumulator[i] = Some(new_acc);
                    }
                    None => {
                        grad_accumulator[i] = Some(scaled_grad);
                    }
                }
            }
        }

        // Optimizer Step (Sync)
        if (step + 1) % accumulation_steps == 0 {
            // Convert Accumulator to Vec<Tensor> (filling None with zeros if needed? or SFO handles missing?)
            // My SFO expects Vec<Tensor> strictly matching vars?
            // "if let Some(grad) = grads.get(i)" in SFO.
            // So we need a Vec<Tensor>. But grad_accumulator is Vec<Option<Tensor>>.
            // We need to compact or pass Option?
            // My SFO implementation: `grads: &Vec<Tensor>`. It iterates vars and uses `grads.get(i)`.
            // Wait, `grads.get(i)` is standard Vec indexing.
            // If accumulation has None, we can't put it in `Vec<Tensor>` easily if we want 1-to-1 mapping by index.
            // SFO impl check:
            // "for (i, var) in self.vars.iter().enumerate() { if let Some(grad) = grads.get(i) ..."
            // Getting index i from Vec<Tensor> works.
            // But if `grad_accumulator` has Nones, we can't make a simple Vec<Tensor> unless we fill with Dummy?
            // BETTER: Update SFO to take `&Vec<Option<Tensor>>`.
            // FOR NOW: I will just unwrap or zero.
            // Actually, if a var has no gradient, it shouldn't update.
            // I will modify `accumulation_to_dense` logic.
            // If None, create a Zero tensor? Expensive.
            // Since this is 8GB constrained, creating Zeros is bad.
            // I should update SFO signature to `Vec<Option<Tensor>>` ideally.
            // But I cannot modify SFO easily in parallel.
            // Hack: SFO `grads.get(i)` returns the element.
            // If I pass a `Vec<Tensor>`, it expects `grads[i]` to correspond to `vars[i]`.
            // If `optim_vars[i]` had no gradient, `grad_accumulator[i]` is None.
            // I cannot skip index in `Vec<Tensor>` without shifting.
            // So I MUST fill gaps.
            // OR... I pass a `Vec` where missing are Zero tensors.
            // This is safer.
            // "grad_accumulator" -> "final_grads".
            let mut final_grads = Vec::with_capacity(optim_vars.len());
            for (i, opt) in grad_accumulator.iter().enumerate() {
                match opt {
                    Some(t) => final_grads.push(t.clone()),
                    None => {
                        // Create zero tensor matching var shape
                        let shape = optim_vars[i].shape();
                        final_grads.push(Tensor::zeros(shape, DType::F32, &device)?);
                    }
                }
            }

            optimizer.step(&final_grads)?;

            // Clear Accumulator
            for item in grad_accumulator.iter_mut() {
                *item = None;
            }
        }

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

            // Debug VRAM
            if let Ok((free, total)) = cortex_rust::device_utils::get_vram_info(0) {
                let used_mb = (total - free) as f64 / 1024.0 / 1024.0;
                info!("       [VRAM] Used: {:.2} MB / Total: {:.2} MB", used_mb, total as f64 / 1024.0 / 1024.0);
            }

            if args.benchmark {
                continue;
            }

            if step % 200 == 0 && step > 0 && val < best_loss {
                best_loss = val;
                info!(
                    "🏆 New best loss: {:.4} (Step {}) - Saving model_best.safetensors",
                    val, step
                );
                save_securely(
                    &varmap,
                    &format!("{}model_best.safetensors", effective_output_dir),
                )?;
                save_training_state(&effective_output_dir, "model_best", step, val)?;
            }
        }

        let stop_path = Path::new("stop_signal");
        if stop_path.exists() {
            info!("\n🛑 Stop signal detected! Saving and exiting...");
            let _ = std::fs::remove_file("stop_signal");
            save_securely(
                &varmap,
                &format!("{}bit_llama_checkpoint.safetensors", effective_output_dir),
            )?;
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
            save_securely(&varmap, &safetensors_path)?;
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

    if let Some(ref output_dir) = args.output_dir {
        let dir = if output_dir.ends_with('/') || output_dir.ends_with('\\') {
            output_dir.clone()
        } else {
            format!("{}/", output_dir)
        };
        std::fs::create_dir_all(output_dir)?;

        let model_path = format!("{}model.safetensors", dir);
        save_securely(&varmap, &model_path)?;
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
        save_securely(&varmap, &format!("{}bit_llama_v1.safetensors", base_dir))?;
    }

    Ok(())
}
