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
    min_lr: f64,
    warmup_steps: usize,
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
            min_lr: 0.00001,   // 最小LR
            warmup_steps: 500, // ウォームアップ
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
        // 1. Mutexのロック処理とトリミングを CentralPanel の前に出す
        // これにより、eguiのパネル内での self 借用とロックが競合するのを防ぐ
        let logs_to_display = {
            let mut logs_guard = self.logs.lock().unwrap();
            let len = logs_guard.len(); // ← 追加

            if len > 50000 {
                let tail = logs_guard.split_off(len - 40000);
                *logs_guard = tail;
            }

            logs_guard.clone()
        };

        // 2. プロセスの終了チェック
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

            // 1. チェックポイント選択
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
                ui.label("Min LR (End):");
                ui.add(egui::Slider::new(&mut self.min_lr, 0.000001..=0.001).logarithmic(true));
            });

            ui.horizontal(|ui| {
                ui.label("Warmup Steps:");
                ui.add(egui::DragValue::new(&mut self.warmup_steps).speed(10));
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
            egui::ScrollArea::vertical()
                .max_height(200.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    ui.code(logs_to_display.as_str());
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
        let min_lr_str = format!("{}", self.min_lr);
        let warmup_str = format!("{}", self.warmup_steps);
        let steps_str = format!("{}", self.steps);
        let save_interval_str = format!("{}", self.save_interval);

        let mut args = vec![
            "run",
            "--release",
            "--features",
            "cuda",
            "--bin",
            "train_llama",
            "--",
            "--lr",
            &lr_str,
            "--min-lr",
            &min_lr_str,
            "--warmup-steps",
            &warmup_str,
            "--steps",
            &steps_str,
            "--save-interval",
            &save_interval_str,
            "--data",
            &self.data_path,
        ];

        // 🚨 Fix: Pass checkpoint path to the trainer
        if let Some(path) = &self.checkpoint_path {
            args.push("--load");
            args.push(path);
        }

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
        // "stop_signal" という空のファイルを作成する
        // これが「止まれ」の合図になります
        match std::fs::File::create("stop_signal") {
            Ok(_) => {
                let mut logs = self.logs.lock().unwrap();
                logs.push_str("\n🛑 Stop signal sent (File created). Waiting for trainer to save and exit...\n");
            }
            Err(e) => {
                let mut logs = self.logs.lock().unwrap();
                logs.push_str(&format!("\n❌ Failed to create stop signal: {}\n", e));
            }
        }
    }
}
