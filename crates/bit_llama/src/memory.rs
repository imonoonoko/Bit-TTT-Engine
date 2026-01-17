use anyhow::{Context, Result};
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug)]
pub struct MemoryEntry {
    pub role: String,
    pub text: String,
    pub timestamp: String,
}

pub struct MemorySystem;

impl MemorySystem {
    pub fn get_memory_dir() -> PathBuf {
        PathBuf::from("workspace").join("memories")
    }

    /// Appends a log entry to workspace/memories/YYYY-MM-DD.jsonl
    pub fn append_log(role: &str, text: &str) -> Result<()> {
        let dir = Self::get_memory_dir();
        fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create memory directory at {:?}", dir))?;

        let today = Local::now().format("%Y-%m-%d").to_string();
        let filename = format!("{}.jsonl", today);
        let path = dir.join(filename);

        let entry = MemoryEntry {
            role: role.to_string(),
            text: text.to_string(),
            timestamp: Local::now().to_rfc3339(),
        };

        let json = serde_json::to_string(&entry)?;

        // Open file in append mode. Create if not exists.
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("Failed to open memory file at {:?}", path))?;

        writeln!(file, "{}", json)?;
        Ok(())
    }

    /// Reads random memories and constructs a training batch string
    pub fn get_replay_batch(max_files: usize) -> Result<String> {
        let dir = Self::get_memory_dir();
        if !dir.exists() {
            return Ok(String::new());
        }

        // 1. Gather all JSONL files
        let mut paths: Vec<PathBuf> = glob::glob(&format!("{}/*.jsonl", dir.display()))?
            .filter_map(Result::ok)
            .collect();

        if paths.is_empty() {
            return Ok(String::new());
        }

        // 2. Shuffle and take random selection
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        paths.shuffle(&mut rng);
        let selected_paths: Vec<_> = paths.into_iter().take(max_files).collect();

        let mut batch_text = String::new();

        // 3. Read and format
        for path in selected_paths {
            let content = fs::read_to_string(&path)?;
            for line in content.lines() {
                if let Ok(entry) = serde_json::from_str::<MemoryEntry>(line) {
                    // Primitive Chat Format for TTT
                    // Ideally we should match the prompt template used in inference.
                    // But raw concatenation "Role: Text" is a good baseline.
                    let prefix = if entry.role == "user" {
                        "\nUser: "
                    } else {
                        "\nAssistant: "
                    };
                    batch_text.push_str(prefix);
                    batch_text.push_str(&entry.text);
                }
            }
            batch_text.push('\n');
        }

        Ok(batch_text)
    }
}
