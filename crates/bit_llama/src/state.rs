//! Bit-Llama Application State
//!
//! Handles runtime state, process management, and logging.

use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::config::ProjectConfig;
use crate::data::concat::Concatenator;

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

    // Logging (Channel-based)
    pub logs: VecDeque<String>,
    pub log_tx: Sender<String>,
    pub log_rx: Receiver<String>,

    pub status_message: String,
    // Async
    pub download_progress: Arc<Mutex<f32>>,
    pub download_status: Arc<Mutex<String>>,

    // UI Cache (Not persisted)
    pub matched_file_count: Option<usize>,
    pub fast_vocab: bool,

    // Async Control
    pub concat_cancel_flag: Arc<AtomicBool>,
}

// Shared State (for UI updates)
pub struct SharedState {
    pub logs: Vec<String>, // Changed from Vec<LogMessage> to Vec<String> to match ProjectState's logs
    pub is_training: bool,
    pub progress: f32, // 0.0 to 1.0
    pub current_step: usize,
    pub total_steps: usize,
    pub loss_history: Vec<[f64; 2]>,    // [step, loss]
    pub vram_usage: Option<(u64, u64)>, // (used_mb, total_mb) // Added
}

impl Default for SharedState {
    fn default() -> Self {
        Self {
            logs: Vec::new(),
            is_training: false,
            progress: 0.0,
            current_step: 0,
            total_steps: 0,
            loss_history: Vec::new(),
            vram_usage: None, // Added
        }
    }
}

impl ProjectState {
    pub fn new(path: PathBuf, config: ProjectConfig) -> Self {
        let (tx, rx) = channel();

        let mut state = Self {
            path,
            config,
            has_corpus: false,
            has_tokenizer: false,
            has_dataset: false,
            active_process: None,
            is_running: false,
            logs: VecDeque::new(),
            log_tx: tx,
            log_rx: rx,
            status_message: "Ready".to_string(),
            download_progress: Arc::new(Mutex::new(0.0)),
            download_status: Arc::new(Mutex::new(String::new())),
            matched_file_count: None,
            fast_vocab: true, // Default to optimized training
            concat_cancel_flag: Arc::new(AtomicBool::new(false)),
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
        // Send to channel (non-blocking)
        let _ = self.log_tx.send(msg.to_string());
    }

    /// Drains the channel and updates the local log buffer.
    /// Should be called from the main UI thread.
    pub fn drain_logs(&mut self) {
        while let Ok(msg) = self.log_rx.try_recv() {
            if msg == "<<PREPROCESS_DONE>>" || msg == "<<CONCAT_DONE>>" {
                self.is_running = false;
                self.status_message = "Ready".to_string();
                self.check_files(); // Refresh file status
                continue;
            }

            self.logs.push_back(msg);
            if self.logs.len() > 1000 {
                self.logs.pop_front();
            }
        }
    }

    /// Drains logs and extracts (step, loss) pairs for graphing.
    /// Returns a vector of extracted data points.
    pub fn drain_logs_with_parse(&mut self) -> Vec<(f64, f64)> {
        let mut data_points = Vec::new();

        while let Ok(msg) = self.log_rx.try_recv() {
            // Check for completion signal
            if msg.contains("<<PREPROCESS_DONE>>") || msg.contains("<<CONCAT_DONE>>") {
                self.is_running = false;
                self.status_message = "Ready".to_string();
                self.check_files();
            }

            // Try to extract step and loss from log line
            if let Some((step, loss)) = Self::parse_training_log(&msg) {
                data_points.push((step, loss));
            }

            self.logs.push_back(msg);
            if self.logs.len() > 1000 {
                self.logs.pop_front();
            }
        }

        data_points
    }

    /// Parse a training log line to extract step and loss.
    fn parse_training_log(line: &str) -> Option<(f64, f64)> {
        let line_lower = line.to_lowercase();

        // Try to find step number
        let step = if let Some(pos) = line_lower.find("step") {
            let after_step = &line[pos + 4..];
            after_step
                .chars()
                .skip_while(|c| !c.is_ascii_digit())
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse::<f64>()
                .ok()
        } else {
            None
        };

        // Try to find loss value
        let loss = if let Some(pos) = line_lower.find("loss") {
            let after_loss = &line[pos + 4..];
            after_loss
                .chars()
                .skip_while(|c| !c.is_ascii_digit() && *c != '.')
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect::<String>()
                .parse::<f64>()
                .ok()
        } else {
            None
        };

        match (step, loss) {
            (Some(s), Some(l)) => Some((s, l)),
            _ => None,
        }
    }

    pub fn get_logs(&self) -> String {
        self.logs.iter().cloned().collect::<Vec<_>>().join("\n")
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

                let tx1 = self.log_tx.clone();
                let tx2 = self.log_tx.clone();

                thread::spawn(move || {
                    let reader = BufReader::new(stdout);
                    for line in reader.lines() {
                        if let Ok(l) = line {
                            let _ = tx1.send(l);
                        }
                    }
                });

                thread::spawn(move || {
                    let reader = BufReader::new(stderr);
                    for line in reader.lines() {
                        if let Ok(l) = line {
                            let _ = tx2.send(l);
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

    pub fn request_stop(&mut self) {
        self.log("🛑 Requesting graceful stop...");
        if let Ok(mut file) = fs::File::create("stop_signal") {
            let _ = file.write_all(b"stop");
            self.log("Signal sent. Waiting for model save...");
        } else {
            self.log("Failed to create stop signal file!");
        }
    }

    pub fn kill_process(&mut self) {
        if let Some(mut child) = self.active_process.take() {
            let _ = child.kill();
            self.log("Process killed by user (Force).");
        }
        self.is_running = false;
        let _ = fs::remove_file("stop_signal");
    }

    pub fn concat_txt_files(&mut self) {
        self.log("Starting corpus concatenation (Async)...");
        self.is_running = true;
        self.status_message = "Concatenating files...".to_string();

        let raw_dir = self.path.join("raw");
        if !raw_dir.exists() {
            self.log(&format!("❌ 'raw' directory not found at: {:?}", raw_dir));
            self.is_running = false;
            return;
        }

        let output_path = self.path.join("data/corpus.txt");
        let raw_str = raw_dir.to_string_lossy().replace("\\", "/");
        let pattern = format!("{}/**/*", raw_str);

        // Reset Cancel Flag
        self.concat_cancel_flag.store(false, Ordering::SeqCst);

        Concatenator::new(
            pattern,
            output_path,
            self.concat_cancel_flag.clone(),
            self.log_tx.clone(),
        )
        .run();
    }

    pub fn cancel_concat(&self) {
        self.concat_cancel_flag.store(true, Ordering::SeqCst);
        self.log("Signal sent: Cancelling concatenation...");
    }
}
