//! Checkpoint Management - Training state persistence

use anyhow::Result;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Training state for serialization
#[derive(serde::Serialize, serde::Deserialize)]
pub struct TrainingState {
    pub step: usize,
    pub loss: f32,
    #[allow(dead_code)]
    pub date: String,
    #[allow(dead_code)]
    pub checkpoint: String,
}

/// Save training state alongside checkpoint file
pub fn save_training_state(
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

    // Also save generic training_state.json for easy resume
    let generic_path = format!("{}training_state.json", base_dir);
    if let Ok(file) = File::create(&generic_path) {
        let _ = serde_json::to_writer_pretty(file, &state);
    }

    Ok(())
}

/// Load training state from JSON file and return the start step
pub fn load_start_step(base_dir: &str) -> usize {
    let state_path = format!("{}training_state.json", base_dir);
    if Path::new(&state_path).exists() {
        if let Ok(file) = File::open(&state_path) {
            let reader = BufReader::new(file);
            if let Ok(json) = serde_json::from_reader::<_, serde_json::Value>(reader) {
                if let Some(s) = json.get("step").and_then(|v| v.as_u64()) {
                    return s as usize;
                }
            }
        }
    }
    0
}

/// Find checkpoint path for loading
pub fn find_checkpoint_path(args_load: Option<&String>, base_dir: &str) -> Option<String> {
    if let Some(path) = args_load {
        tracing::info!("📂 Loading specific checkpoint from Launcher: {}", path);
        return Some(path.clone());
    }

    if Path::new("bit_llama_checkpoint.safetensors").exists() {
        return Some("bit_llama_checkpoint.safetensors".to_string());
    }

    if Path::new("bit_llama/bit_llama_checkpoint.safetensors").exists() {
        return Some("bit_llama/bit_llama_checkpoint.safetensors".to_string());
    }

    let base_checkpoint = format!("{}bit_llama_checkpoint.safetensors", base_dir);
    if Path::new(&base_checkpoint).exists() {
        return Some(base_checkpoint);
    }

    None
}
