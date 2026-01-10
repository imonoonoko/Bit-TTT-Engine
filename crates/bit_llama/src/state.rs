//! Bit-Llama Application State
//!
//! Handles runtime state, process management, and logging.

use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::config::ProjectConfig;

// Runtime State (Not saved to disk)
pub struct ProjectState {
    pub path: PathBuf,
    pub config: ProjectConfig,
    // Status
    pub has_corpus: bool,
    pub has_tokenizer: bool,
    pub has_dataset: bool,
    // Processes
    pub active_process: Option<Child>,
    pub is_running: bool,
    pub logs: Arc<Mutex<VecDeque<String>>>,
    pub status_message: String,
    // Async
    pub download_progress: Arc<Mutex<f32>>,
    pub download_status: Arc<Mutex<String>>,
}

impl ProjectState {
    pub fn new(path: PathBuf, config: ProjectConfig) -> Self {
        let mut state = Self {
            path,
            config,
            has_corpus: false,
            has_tokenizer: false,
            has_dataset: false,
            active_process: None,
            is_running: false,
            logs: Arc::new(Mutex::new(VecDeque::new())),
            status_message: "Ready".to_string(),
            download_progress: Arc::new(Mutex::new(0.0)),
            download_status: Arc::new(Mutex::new(String::new())),
        };
        state.check_files();
        state
    }

    pub fn check_files(&mut self) {
        self.has_corpus = self.path.join("data/corpus.txt").exists();
        self.has_tokenizer = self.path.join("data/tokenizer.json").exists();
        self.has_dataset = self.path.join("data/train.u32").exists();
    }

    pub fn save_config(&self) {
        let config_path = self.path.join("config.json");
        if let Ok(json) = serde_json::to_string_pretty(&self.config) {
            if let Err(e) = fs::write(&config_path, json) {
                self.log(&format!("Failed to save config: {}", e));
            } else {
                self.log("✅ Configuration saved.");
            }
        }
    }

    pub fn log(&self, msg: &str) {
        let mut logs = self.logs.lock().unwrap();
        logs.push_back(msg.to_string());
        if logs.len() > 1000 {
            logs.pop_front();
        }
    }

    pub fn get_logs(&self) -> String {
        let logs = self.logs.lock().unwrap();
        logs.iter().cloned().collect::<Vec<_>>().join("\n")
    }

    pub fn run_command(&mut self, cmd: &str, args: &[&str]) {
        self.is_running = true;
        self.status_message = format!("Running {}...", cmd);
        self.log(&format!("$ {} {}", cmd, args.join(" ")));

        let mut command = Command::new(cmd);
        command
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        match command.spawn() {
            Ok(mut child) => {
                let stdout = child.stdout.take().unwrap();
                let stderr = child.stderr.take().unwrap();
                let logs1 = self.logs.clone();
                let logs2 = self.logs.clone();

                thread::spawn(move || {
                    let reader = BufReader::new(stdout);
                    for line in reader.lines() {
                        if let Ok(l) = line {
                            let mut logs = logs1.lock().unwrap();
                            logs.push_back(l);
                            if logs.len() > 1000 {
                                logs.pop_front();
                            }
                        }
                    }
                });

                thread::spawn(move || {
                    let reader = BufReader::new(stderr);
                    for line in reader.lines() {
                        if let Ok(l) = line {
                            let mut logs = logs2.lock().unwrap();
                            logs.push_back(l);
                            if logs.len() > 1000 {
                                logs.pop_front();
                            }
                        }
                    }
                });

                self.active_process = Some(child);
            }
            Err(e) => {
                self.log(&format!("Failed to start: {}", e));
                self.is_running = false;
            }
        }
    }

    pub fn stop_process(&mut self) {
        if let Some(mut child) = self.active_process.take() {
            let _ = child.kill();
            self.log("Process killed by user.");
        }
        self.is_running = false;
    }

    pub fn concat_txt_files(&mut self) {
        self.log("Starting corpus concatenation...");

        let raw_dir = self.path.join("raw");
        let output_path = self.path.join("data/corpus.txt");

        let mut count = 0;
        let mut total_bytes = 0;

        if let Ok(entries) = fs::read_dir(&raw_dir) {
            if let Ok(mut out_file) = fs::File::create(&output_path) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().map_or(false, |ext| ext == "txt") {
                        if let Ok(content) = fs::read(&path) {
                            if let Err(e) = out_file.write_all(&content) {
                                self.log(&format!("Write error: {}", e));
                            }
                            let _ = out_file.write_all(b"\n");
                            count += 1;
                            total_bytes += content.len();
                        }
                    }
                }
            }
        }

        self.log(&format!(
            "✅ Concatenated {} files in {:.2} MB.",
            count,
            total_bytes as f64 / 1_048_576.0
        ));
        self.check_files();
    }
}
