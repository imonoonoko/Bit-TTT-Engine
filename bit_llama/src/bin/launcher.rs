//! Bit-TTT Training Launcher GUI
//!
//! A simple GUI for controlling the training process.
//! Run with: cargo run --bin launcher

use eframe::egui;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([600.0, 500.0])
            .with_title("🚀 Bit-TTT Training Launcher"),
        ..Default::default()
    };
    eframe::run_native(
        "Bit-TTT Trainer Launcher",
        options,
        Box::new(|_cc| Box::new(MyApp::default())),
    )
}

struct MyApp {
    lr: f64,
    steps: usize,
    save_interval: usize,
    checkpoint_path: Option<String>,
    data_path: String,
    logs: Arc<Mutex<String>>,
    process: Option<Child>,
    is_running: Arc<Mutex<bool>>,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            lr: 0.00005,
            steps: 10000,
            save_interval: 1000,
            checkpoint_path: None,
            data_path: "data/TinyStories/train.bin".to_string(),
            logs: Arc::new(Mutex::new(String::new())),
            process: None,
            is_running: Arc::new(Mutex::new(false)),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Check if process finished
        if let Some(ref mut child) = self.process {
            match child.try_wait() {
                Ok(Some(_)) => {
                    self.process = None;
                    *self.is_running.lock().unwrap() = false;
                    let mut logs = self.logs.lock().unwrap();
                    logs.push_str("\n✅ Training process finished.\n");
                }
                Ok(None) => {}
                Err(e) => {
                    let mut logs = self.logs.lock().unwrap();
                    logs.push_str(&format!("\n❌ Error: {}\n", e));
                }
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🚀 Bit-TTT Training Control");
            ui.separator();

            // 1. Checkpoint selection
            ui.horizontal(|ui| {
                ui.label("Checkpoint:");
                if let Some(path) = &self.checkpoint_path {
                    ui.monospace(path);
                } else {
                    ui.label("(Start Fresh)");
                }
                if ui.button("📂 Load...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("SafeTensors", &["safetensors"])
                        .pick_file()
                    {
                        self.checkpoint_path = Some(path.display().to_string());
                    }
                }
                if ui.button("🗑 Clear").clicked() {
                    self.checkpoint_path = None;
                }
            });

            ui.separator();

            // 2. hyperparameters
            ui.heading("⚙️ Training Settings");

            ui.horizontal(|ui| {
                ui.label("Learning Rate:");
                ui.add(egui::Slider::new(&mut self.lr, 0.00001..=0.01).logarithmic(true));
            });

            ui.horizontal(|ui| {
                ui.label("Total Steps:");
                ui.add(egui::DragValue::new(&mut self.steps).speed(100));
            });

            ui.horizontal(|ui| {
                ui.label("Save Interval:");
                ui.add(egui::DragValue::new(&mut self.save_interval).speed(100));
            });

            ui.horizontal(|ui| {
                ui.label("Data Path:");
                ui.text_edit_singleline(&mut self.data_path);
            });

            ui.separator();

            // 3. Start/Stop buttons
            ui.horizontal(|ui| {
                let is_running = self.process.is_some();

                if !is_running {
                    if ui.button("▶ START Training").clicked() {
                        self.start_training();
                    }
                } else {
                    // Show stop button with spinner
                    if ui.button("⏹ STOP & SAVE").clicked() {
                        self.stop_training();
                    }
                    ui.spinner();
                    ui.label("Training in progress...");
                }

                if ui.button("🗑 Clear Logs").clicked() {
                    self.logs.lock().unwrap().clear();
                }
            });

            ui.separator();

            // 4. Log display area
            ui.heading("📋 Logs:");
            let logs = self.logs.lock().unwrap();
            egui::ScrollArea::vertical()
                .max_height(200.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut logs.as_str())
                            .font(egui::TextStyle::Monospace)
                            .desired_width(f32::INFINITY),
                    );
                });

            // Request repaint while running to update logs
            if self.process.is_some() {
                ctx.request_repaint();
            }
        });
    }
}

impl MyApp {
    fn start_training(&mut self) {
        // Clear old logs
        self.logs.lock().unwrap().clear();

        let lr_str = format!("{}", self.lr);
        let steps_str = format!("{}", self.steps);
        let save_interval_str = format!("{}", self.save_interval);

        let args = vec![
            "run",
            "--release",
            "--features",
            "cuda",
            "--bin",
            "train_llama",
            "--",
            "--lr",
            &lr_str,
            "--steps",
            &steps_str,
            "--save-interval",
            &save_interval_str,
            "--data",
            &self.data_path,
        ];

        // Add checkpoint path if specified (future: implement --load flag)
        let _checkpoint = self.checkpoint_path.clone();

        {
            let mut logs = self.logs.lock().unwrap();
            logs.push_str(&format!(
                "🚀 Starting training...\n   LR: {}\n   Steps: {}\n   Save Interval: {}\n   Data: {}\n\n",
                self.lr, self.steps, self.save_interval, self.data_path
            ));
        }

        // Spawn cargo process
        match Command::new("cargo")
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(mut child) => {
                *self.is_running.lock().unwrap() = true;

                // Read stdout in background thread
                if let Some(stdout) = child.stdout.take() {
                    let logs_clone = self.logs.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stdout);
                        for line in reader.lines() {
                            if let Ok(l) = line {
                                let mut logs = logs_clone.lock().unwrap();
                                logs.push_str(&l);
                                logs.push('\n');
                            }
                        }
                    });
                }

                // Read stderr in background thread
                if let Some(stderr) = child.stderr.take() {
                    let logs_clone = self.logs.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stderr);
                        for line in reader.lines() {
                            if let Ok(l) = line {
                                let mut logs = logs_clone.lock().unwrap();
                                logs.push_str("[ERR] ");
                                logs.push_str(&l);
                                logs.push('\n');
                            }
                        }
                    });
                }

                self.process = Some(child);
            }
            Err(e) => {
                let mut logs = self.logs.lock().unwrap();
                logs.push_str(&format!("❌ Failed to start: {}\n", e));
            }
        }
    }

    fn stop_training(&mut self) {
        if let Some(ref mut child) = self.process {
            // On Windows, we can't send SIGINT easily, so we use kill
            // The train_llama process has Ctrl+C handling, but from GUI we'll use taskkill
            #[cfg(windows)]
            {
                let pid = child.id();
                // Try to send Ctrl+C signal via taskkill /PID /T (tree kill)
                let _ = Command::new("taskkill")
                    .args(["/PID", &pid.to_string(), "/T"])
                    .output();
            }

            #[cfg(not(windows))]
            {
                let _ = child.kill();
            }

            let mut logs = self.logs.lock().unwrap();
            logs.push_str("\n🛑 Stop signal sent. Waiting for graceful shutdown...\n");
        }
    }
}
