use crate::chat::{Message, Role};
use cortex_rust::Llama;
use std::io::{self, Write};
use std::path::Path;

pub fn run_chat(
    path_str: &str,
    temp: f64,
    max_tokens: usize,
    system_prompt: &str,
) -> anyhow::Result<()> {
    println!("🚀 Initializing Ore-Llama Engine...");

    let path = Path::new(path_str);

    println!("📂 Loading model from: {:?}", path);
    let mut llama = Llama::load_auto(path)?;

    println!("✅ Model Loaded!");
    println!("Type '/bye' to exit.\n");
    println!("System: {}", system_prompt);

    let mut history = vec![Message::new(Role::System, system_prompt.to_string())];

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }
        if input == "/bye" || input == "/exit" {
            break;
        }

        history.push(Message::new(Role::User, input.to_string()));

        // Build prompt from history
        let prompt_to_send = history
            .iter()
            .map(|m| m.to_prompt_line())
            .collect::<String>()
            + "AI: ";

        print!("🤖: ");
        io::stdout().flush()?;

        let generated_text =
            llama.stream_completion(&prompt_to_send, max_tokens, temp, |token| {
                print!("{}", token);
                io::stdout().flush()?;
                Ok(true)
            })?;

        println!(); // new line after generation
        history.push(Message::new(Role::AI, generated_text));
    }

    println!("👋 Bye!");
    Ok(())
}

pub fn list_models() -> anyhow::Result<()> {
    println!("🔍 Scanning for models...");
    let dirs = vec![Path::new("."), Path::new("models")];

    for dir in dirs {
        if !dir.exists() {
            continue;
        }

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext == "bitt" {
                            println!("  📦 {}", path.display());
                        }
                    }
                } else if path.is_dir() {
                    // Check for config.json
                    if path.join("config.json").exists() && path.join("model.safetensors").exists()
                    {
                        println!("  📂 {} (Legacy Dir)", path.display());
                    }
                }
            }
        }
    }
    Ok(())
}
