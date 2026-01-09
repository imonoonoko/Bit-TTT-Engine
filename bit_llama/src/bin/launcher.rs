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

struct TrainingStatus {
    step: usize,
    total_steps: usize,
    loss: f32,
    lr: f64,
    message: String,
}

#[derive(serde::Deserialize)]
struct TrainingState {
    step: usize,
    loss: f32,
    #[allow(dead_code)]
    date: String,
    #[allow(dead_code)]
    checkpoint: String,
}

impl Default for TrainingStatus {
    fn default() -> Self {
        Self {
            step: 0,
            total_steps: 10000,
            loss: 0.0,
            lr: 0.0,
            message: "Ready to start".to_string(),
        }
    }
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
    status: Arc<Mutex<TrainingStatus>>, // 新機能: 進捗状態管理
    show_logs: bool,                    // 新機能: ログ表示切り替え
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
            status: Arc::new(Mutex::new(TrainingStatus::default())),
            show_logs: false, // デフォルトはログ非表示
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
            let len = logs_guard.len();

            // 定期的なトリミング (メモリ節約)
            if len > 100000 {
                let tail = logs_guard.split_off(len - 80000);
                *logs_guard = tail;
            }

            // GUI表示用には末尾のみをコピーする (GUIフリーズ防止)
            // 毎回数万文字をcloneすると重いため、直近5000文字程度に制限
            let display_limit = 5000;
            if logs_guard.len() > display_limit {
                logs_guard[logs_guard.len() - display_limit..].to_string()
            } else {
                logs_guard.clone()
            }
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
                    let dialog = rfd::FileDialog::new().add_filter("SafeTensors", &["safetensors"]);

                    // Set default path to bit_llama directory if it exists
                    let dialog = if let Ok(cwd) = std::env::current_dir() {
                        let bit_llama_dir = cwd.join("bit_llama");
                        if bit_llama_dir.exists() {
                            dialog.set_directory(&bit_llama_dir)
                        } else {
                            dialog
                        }
                    } else {
                        dialog
                    };

                    if let Some(path_buf) = dialog.pick_file() {
                        let path_str = path_buf.display().to_string();
                        self.checkpoint_path = Some(path_str.clone());

                        // Try to read associated JSON metadata
                        // If path is "model.safetensors", look for "model.json"
                        let json_path = path_buf.with_extension("json");
                        if json_path.exists() {
                            if let Ok(file) = std::fs::File::open(&json_path) {
                                let reader = std::io::BufReader::new(file);
                                let state_result: serde_json::Result<TrainingState> =
                                    serde_json::from_reader(reader);
                                if let Ok(state) = state_result {
                                    let mut status = self.status.lock().unwrap();
                                    status.step = state.step;
                                    status.loss = state.loss;
                                    status.message = format!(
                                        "Set to resume from Step {} (loaded from JSON)",
                                        state.step
                                    );

                                    // Also update step input in GUI
                                    if state.step < self.steps {
                                        // Keep total steps as is unless it's smaller than current
                                    }
                                }
                            }
                        }
                    }
                }
                if ui.button("🗑 Clear").clicked() {
                    self.checkpoint_path = None;
                    // Reset status to 0
                    let mut status = self.status.lock().unwrap();
                    status.step = 0;
                    status.loss = 0.0;
                    status.message = "Ready to start".to_string();
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

            // 4. Progress Dashboard (New!)
            ui.separator();
            ui.heading("📊 Progress:");

            let status = self.status.lock().unwrap();
            let progress = if status.total_steps > 0 {
                status.step as f32 / status.total_steps as f32
            } else {
                0.0
            };

            // A. Metrics Grid
            egui::Grid::new("metrics_grid")
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Current Step:");
                    ui.label(format!("{} / {}", status.step, status.total_steps));
                    ui.end_row();

                    ui.label("Loss:");
                    ui.label(
                        egui::RichText::new(format!("{:.4}", status.loss))
                            .strong()
                            .color(egui::Color32::LIGHT_RED),
                    );
                    ui.end_row();

                    ui.label("Learning Rate:");
                    ui.label(format!("{:.7}", status.lr));
                    ui.end_row();
                });

            ui.add_space(5.0);

            // B. Progress Bar
            ui.add(
                egui::ProgressBar::new(progress)
                    .show_percentage()
                    .animate(true),
            );
            ui.label(egui::RichText::new(&status.message).italics().weak()); // show last log line

            ui.separator();

            // 5. Log display (Optional)
            if ui
                .checkbox(
                    &mut self.show_logs,
                    "Show Full Logs (May affect performance)",
                )
                .changed()
            {
                // Toggle logic if needed
            }

            if self.show_logs {
                ui.heading("📋 Logs:");
                egui::ScrollArea::vertical()
                    .max_height(200.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        ui.code(logs_to_display.as_str());
                    });
            }

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

        // ★ステータス（プログレスバーなど）のリセット
        {
            let mut status = self.status.lock().unwrap();
            status.step = 0;
            status.loss = 0.0;
            status.message = "Starting...".to_string();
            // ※Checkpointロード時は、ログから "Resuming..." が来るので自動で更新されます
        }

        // 🛡️ Safety: Ensure no leftover stop signal exists
        let _ = std::fs::remove_file("stop_signal");

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
                    let status_clone = self.status.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stdout);
                        for line in reader.lines() {
                            if let Ok(l) = line {
                                // 1. ログ保存
                                {
                                    let mut logs = logs_clone.lock().unwrap();
                                    logs.push_str(&l);
                                    logs.push('\n');
                                }

                                // 2. ステータス解析 (Extracted function)
                                let mut status = status_clone.lock().unwrap();
                                parse_log_line(&l, &mut status);
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
        if let Ok(path) = std::env::current_dir() {
            let signal_path = path.join("stop_signal");
            match std::fs::File::create(&signal_path) {
                Ok(_) => {
                    let mut logs = self.logs.lock().unwrap();
                    logs.push_str(&format!(
                        "\n🛑 Stop signal sent to: {:?}\nWaiting for trainer to save and exit...\n",
                        signal_path
                    ));
                }
                Err(e) => {
                    let mut logs = self.logs.lock().unwrap();
                    logs.push_str(&format!(
                        "\n❌ Failed to create stop signal at {:?}: {}\n",
                        signal_path, e
                    ));
                }
            }
        }
    }
}

// アプリ終了時（ウィンドウを閉じた時）のクリーンアップ処理
impl Drop for MyApp {
    fn drop(&mut self) {
        // プロセスがまだ動いていれば強制終了させる
        if let Some(mut child) = self.process.take() {
            let _ = child.kill(); // 念の為 kill しておく
                                  // ※ Windowsなら kill でOK。stop_signal を待つ余裕がないため強制終了が安全。
        }
    }
}

/// Helper function to parse log lines and update status
fn parse_log_line(line: &str, status: &mut TrainingStatus) {
    // 1. Step取得 (汎用的)
    if let Some(idx) = line.find("|") {
        let prefix = &line[..idx]; // "Step 123 "
        if let Some(step_str) = prefix.trim().strip_prefix("Step ") {
            if let Ok(step_val) = step_str.trim().parse::<usize>() {
                if step_val > status.step {
                    status.step = step_val;
                }
            }
        }
    }

    // 2. Resume検知
    if let Some(idx) = line.find("Resuming from Step ") {
        let remaining = &line[idx + 19..];
        let val_str = remaining.split_whitespace().next().unwrap_or("");
        if let Ok(resume_step) = val_str.parse::<usize>() {
            status.step = resume_step;
            status.message = format!("Resumed from Step {}", resume_step);
        }
    }

    // 3. Loss取得
    if let Some(idx) = line.find("Loss: ") {
        let remaining = &line[idx + 6..];
        let val_str = remaining.split_whitespace().next().unwrap_or("");
        if let Ok(loss_val) = val_str.parse::<f32>() {
            status.loss = loss_val;
        }
    }

    // 4. LR取得
    if let Some(idx) = line.find("LR: ") {
        let remaining = &line[idx + 4..];
        let val_str = remaining.split_whitespace().next().unwrap_or("");
        if let Ok(lr_val) = val_str.parse::<f64>() {
            status.lr = lr_val;
        }
    }

    // 5. 最新メッセージ更新
    if line.len() < 100 {
        status.message = line.to_string();
    }
}
