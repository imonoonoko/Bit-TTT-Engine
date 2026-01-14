use crate::memory::MemorySystem;
use anyhow::Result;
use clap::Args;
use cortex_rust::Llama;
use std::io::{self, Write};
use std::sync::mpsc::channel;
use std::thread;

#[derive(Args, Debug, Clone)]
pub struct InferenceArgs {
    #[arg(short, long, default_value = ".")]
    pub model: String,

    #[arg(long, default_value_t = 100)]
    pub max_tokens: usize,

    #[arg(long, default_value_t = 0.8)]
    pub temp: f64,

    #[arg(short, long)]
    pub prompt: Option<String>,

    /// Path to load initial TTT memory (.soul file)
    #[arg(long)]
    pub memory: Option<String>,
}

pub fn run(args: InferenceArgs) -> Result<()> {
    println!("--- Bit-Llama Inference ---");
    println!("Loading model from: {}", args.model);

    // Ensure "souls" directory exists next to executable
    let exe_path = std::env::current_exe()?;
    let exe_dir = exe_path.parent().unwrap_or(std::path::Path::new("."));
    let souls_dir = exe_dir.join("souls");
    if !souls_dir.exists() {
        std::fs::create_dir_all(&souls_dir).ok();
        println!("Created souls directory: {:?}", souls_dir);
    }

    // Helper to resolve paths relative to souls dir
    let resolve_path = |p: &str| -> std::path::PathBuf {
        let path = std::path::Path::new(p);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            souls_dir.join(p)
        }
    };

    let mut llama = Llama::load_auto(&args.model).map_err(|e| {
        anyhow::anyhow!(
            "Failed to load model: {}\nEnsure directory contains config.json etc.",
            e
        )
    })?;

    llama.model.precompute_packed()?;

    // Load initial memory if specified
    if let Some(mem_path) = &args.memory {
        let path = resolve_path(mem_path);
        println!("Loading memory from: {:?}", path);
        if let Err(e) = llama.load_memory(&path) {
            eprintln!("⚠️ Failed to load memory: {}", e);
        } else {
            println!("✅ Memory (.soul) loaded successfully!");
            println!("🌟 Current Soul Level: {}", llama.soul_level);
        }
    }

    println!("✅ Model Loaded! (Soul Level: {})", llama.soul_level);

    let mut current_temp = args.temp;
    let mut current_max_tokens = args.max_tokens;

    // One-shot mode if prompt provided
    if let Some(p) = &args.prompt {
        println!("\n> {}", p);
        if let Err(e) = MemorySystem::append_log("user", p) {
            eprintln!("(Log Error: {})", e);
        }
        println!("[Generating...]");
        let callback = |token: &str| -> anyhow::Result<bool> {
            print!("{}", token);
            io::stdout().flush()?;
            Ok(true)
        };
        match llama.stream_completion(p, current_max_tokens, current_temp, callback) {
            Ok(full_text) => {
                println!();
                println!("(Soul Level: {})", llama.soul_level);
                let response = if full_text.starts_with(p) {
                    &full_text[p.len()..]
                } else {
                    &full_text
                };
                MemorySystem::append_log("assistant", response.trim()).ok();
            }
            Err(e) => println!("Error: {}", e),
        }
        return Ok(());
    }

    // Interactive Loop - Threaded Input
    let (input_tx, input_rx) = channel();
    thread::spawn(move || loop {
        let mut line = String::new();
        if io::stdin().read_line(&mut line).is_ok() {
            if input_tx.send(line).is_err() {
                break;
            }
        } else {
            break;
        }
    });

    // State variables OUTSIDE loop
    let mut is_sleeping = false;
    let mut sleep_chunks: Vec<String> = Vec::new();
    let mut sleep_index = 0;

    loop {
        // [BLOCK A] Input Handling
        let mut input_cmd: Option<String> = None;

        if is_sleeping {
            // Non-blocking poll for interrupt
            if let Ok(cmd) = input_rx.try_recv() {
                input_cmd = Some(cmd);
            }
        } else {
            // Blocking input
            eprintln!("<<READY>>"); // Signal to GUI via stderr
            print!("\n> "); // Visual prompt to stdout
            io::stdout().flush()?;
            match input_rx.recv() {
                Ok(s) => input_cmd = Some(s),
                Err(_) => break, // EOF
            }
        }

        // [BLOCK B] Command Processing
        if let Some(raw) = input_cmd {
            let prompt = raw.trim().to_string();

            // Handle Interrupts during sleep
            if is_sleeping {
                if prompt == "/wake" {
                    println!("☀ Waking up... Saving progress.");
                    if let Some(mem_path) = &args.memory {
                        let path = resolve_path(mem_path);
                        if let Err(e) = llama.save_memory(&path) {
                            println!("❌ Failed to save soul: {}", e);
                        } else {
                            println!("💾 Soul saved.");
                        }
                    }
                    is_sleeping = false;
                    sleep_chunks.clear();
                    continue;
                } else if prompt == "/quit" || prompt == "exit" {
                    println!("💤 Dream interrupted by exit.");
                    break;
                } else {
                    // Ignore empty or other inputs during sleep, but don't print unless meaningful
                    if !prompt.is_empty() {
                        println!("💤 I'm dreaming... (/wake to interrupt)");
                    }
                }
            } else {
                // Awake Mode Commands
                if prompt.is_empty() {
                    continue;
                }
                if prompt == "/quit" || prompt == "exit" {
                    break;
                }

                if prompt == "/sleep" {
                    println!("🌙 Entering Sleep Mode (Offline Learning)...");
                    match MemorySystem::get_replay_batch(5) {
                        Ok(training_data) => {
                            if training_data.is_empty() {
                                println!("⚠️ No memories found.");
                            } else {
                                println!("🧠 Dreaming... ({} chars)", training_data.len());
                                // Force flush to ensure GUI sees this before loop tightens
                                io::stdout().flush().ok();

                                sleep_chunks = training_data
                                    .chars()
                                    .collect::<Vec<char>>()
                                    .chunks(500)
                                    .map(|c| c.iter().collect())
                                    .collect();
                                sleep_index = 0;
                                is_sleeping = true;
                                continue;
                            }
                        }
                        Err(e) => {
                            println!("❌ Database Error: {}", e);
                        }
                    }
                    continue;
                }

                // ... (Save/Load/Reset logic unchanged, just use println for errors) ...
                if prompt == "/reset" {
                    llama.reset_state()?;
                    println!("🔄 Reset.");
                    continue;
                }

                if let Some(path) = prompt.strip_prefix("/save ") {
                    let mut path_str = path.trim().to_string();
                    if !path_str.contains('.') {
                        path_str.push_str(".soul");
                    }
                    let path = resolve_path(&path_str);

                    if let Err(e) = llama.save_memory(&path) {
                        println!("❌ Failed to save memory: {}", e);
                    } else {
                        println!("💾 Memory saved to: {:?}", path);
                    }
                    continue;
                }

                if let Some(path) = prompt.strip_prefix("/load ") {
                    let path_str = path.trim();
                    let path = resolve_path(path_str);
                    if let Err(e) = llama.load_memory(&path) {
                        println!("❌ Failed to load memory: {}", e);
                    } else {
                        println!("📂 Memory loaded from: {:?}", path);
                        println!("🌟 Current Soul Level: {}", llama.soul_level);
                    }
                    continue;
                }

                if let Some(stripped) = prompt.strip_prefix("/temp ") {
                    if let Ok(v) = stripped.parse::<f64>() {
                        current_temp = v;
                        println!("🌡️ Temperature set to {:.2}", current_temp);
                    } else {
                        println!("❌ Invalid temperature format.");
                    }
                    continue;
                }

                if let Some(stripped) = prompt.strip_prefix("/len ") {
                    if let Ok(v) = stripped.parse::<usize>() {
                        current_max_tokens = v;
                        println!("📏 Max length set to {}", current_max_tokens);
                    } else {
                        println!("❌ Invalid length format.");
                    }
                    continue;
                }

                // Standard Generation
                if !prompt.starts_with("/") {
                    // ... (Generation logic unchanged) ...
                    // Shortened for brevity
                    if let Err(e) = MemorySystem::append_log("user", &prompt) {
                        eprintln!("(Log Error: {})", e);
                    }
                    println!("[Generating...]");
                    let callback = |token: &str| -> anyhow::Result<bool> {
                        print!("{}", token);
                        io::stdout().flush()?;
                        Ok(true)
                    };
                    if let Ok(full) =
                        llama.stream_completion(&prompt, current_max_tokens, current_temp, callback)
                    {
                        println!("\n(Soul Level: {})", llama.soul_level);
                        let resp = if full.starts_with(&prompt) {
                            &full[prompt.len()..]
                        } else {
                            &full
                        };
                        MemorySystem::append_log("assistant", resp.trim()).ok();
                    }
                }
            }
        }

        // [BLOCK C] Automated Sleep Processing
        if is_sleeping {
            if sleep_index < sleep_chunks.len() {
                let chunk = &sleep_chunks[sleep_index];
                match llama.learn(chunk) {
                    Ok(_) => {
                        print!(".");
                        let _ = io::stdout().flush();
                        sleep_index += 1;
                        // Add a sleep for UX "dreaming" effect (Tamagotchi-like pacing)
                        thread::sleep(std::time::Duration::from_millis(100));
                    }
                    Err(e) => {
                        // 【修正】println! を使用してGUIに出す。
                        println!("\n❌ Error during learning: {}", e);
                        is_sleeping = false;
                        sleep_chunks.clear();
                    }
                }
            } else {
                println!("\n✨ Sleep finished.");
                println!("🌟 Soul Level: {}", llama.soul_level);
                if let Some(mem_path) = &args.memory {
                    let path = resolve_path(mem_path);
                    llama.save_memory(&path).ok();
                    println!("💾 Auto-saved.");
                }
                is_sleeping = false;
                sleep_chunks.clear();
            }
        }
    }

    Ok(())
}
