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
        Box::new(|cc| {
            setup_custom_fonts(&cc.egui_ctx);
            Box::new(MyApp::default())
        }),
    )
}

fn setup_custom_fonts(ctx: &egui::Context) {
    // 4. フォント設定 (日本語対応)
    let mut fonts = egui::FontDefinitions::default();

    // プラットフォームごとのフォント検索
    let font_candidates = if cfg!(target_os = "windows") {
        vec![
            "C:/Windows/Fonts/YuGothB.ttc",
            "C:/Windows/Fonts/msgothic.ttc",
            "C:/Windows/Fonts/meiryo.ttc",
        ]
    } else if cfg!(target_os = "macos") {
        vec![
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    } else {
        // Linux / Other
        vec![
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        ]
    };

    let mut font_data: Option<Vec<u8>> = None;
    for path in font_candidates {
        if std::path::Path::new(path).exists() {
            if let Ok(data) = std::fs::read(path) {
                font_data = Some(data);
                break;
            }
        }
    }

    if let Some(data) = font_data {
        fonts
            .font_data
            .insert("jp_font".to_owned(), egui::FontData::from_owned(data));

        // Put my font as first choice for proportional text
        fonts
            .families
            .entry(egui::FontFamily::Proportional)
            .or_default()
            .insert(0, "jp_font".to_owned());

        // Put my font as first choice for monospace text
        fonts
            .families
            .entry(egui::FontFamily::Monospace)
            .or_default()
            .insert(0, "jp_font".to_owned());

        ctx.set_fonts(fonts);
    }
}

struct TrainingStatus {
    step: usize,
    total_steps: usize,
    loss: f32,
    lr: f64,
    message: String,
    is_compiling: bool, // New: shows if cargo is compiling
}

impl Default for TrainingStatus {
    fn default() -> Self {
        Self {
            step: 0,
            total_steps: 10000,
            loss: 0.0,
            lr: 0.0,
            message: "Ready to start".to_string(),
            is_compiling: false,
        }
    }
}

#[derive(PartialEq, Clone, Copy)]
enum Language {
    English,
    Japanese,
}

struct MyApp {
    // Settings
    lr: f64,
    min_lr: f64,
    warmup_steps: usize,
    steps: usize,
    save_interval: usize,
    checkpoint_path: Option<String>,
    data_path: String,

    // UI State
    logs: Arc<Mutex<String>>,
    status: Arc<Mutex<TrainingStatus>>,

    process: Option<Child>,
    is_running: Arc<Mutex<bool>>,
    language: Language,
    use_gpu: bool, // New: GPU Toggle
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            lr: 0.001,          // Default from train_llama.rs
            min_lr: 0.0001,     // Default from train_llama.rs
            warmup_steps: 100,  // Default from train_llama.rs
            steps: 10000,       // Default
            save_interval: 500, // Default from train_llama.rs
            checkpoint_path: None,
            data_path: "data/TinyStories/train.bin".to_string(),
            logs: Arc::new(Mutex::new(String::new())),
            status: Arc::new(Mutex::new(TrainingStatus::default())),

            process: None,
            is_running: Arc::new(Mutex::new(false)),
            language: Language::Japanese, // Default to Japanese as requested
            use_gpu: true,                // Default to GPU
        }
    }
}

impl MyApp {
    /// Helper for localization
    fn text(&self, en: &str, ja: &str) -> String {
        match self.language {
            Language::English => en.to_string(),
            Language::Japanese => ja.to_string(),
        }
    }

    fn start_training(&mut self) {
        // Clear old logs
        self.logs.lock().unwrap().clear();

        // ★ステータス（プログレスバーなど）のリセット
        {
            let mut status = self.status.lock().unwrap();
            status.step = 0;
            status.loss = 0.0;
            status.is_compiling = true; // Show compiling indicator immediately
            status.message = self.text("Starting... (Compiling)", "起動中... (コンパイル中)");
        }

        // 🛡️ Safety: Ensure no leftover stop signal exists
        let _ = std::fs::remove_file("stop_signal");

        let lr_str = format!("{}", self.lr);
        let min_lr_str = format!("{}", self.min_lr);
        let warmup_str = format!("{}", self.warmup_steps);
        let steps_str = format!("{}", self.steps);
        let save_interval_str = format!("{}", self.save_interval);

        let mut args = vec!["run", "--release"];

        // Conditionally enable CUDA feature
        if self.use_gpu {
            args.push("--features");
            args.push("cuda");
        }

        args.push("--bin");
        args.push("train_llama");
        args.push("--");

        args.extend_from_slice(&[
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
        ]);

        // 🚨 Pass checkpoint path to the trainer
        if let Some(path) = &self.checkpoint_path {
            args.push("--load");
            args.push(path);
        }

        {
            let mut logs = self.logs.lock().unwrap();
            logs.push_str(&format!(
                "🚀 {}...\n   LR: {}\n   Steps: {}\n   Save Interval: {}\n   Data: {}\n\n",
                self.text("Starting Training", "トレーニングを開始します"),
                self.lr,
                self.steps,
                self.save_interval,
                self.data_path
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
                                {
                                    let mut logs = logs_clone.lock().unwrap();
                                    logs.push_str(&l);
                                    logs.push('\n');
                                }
                                let mut status = status_clone.lock().unwrap();
                                parse_log_line(&l, &mut status);
                            }
                        }
                    });
                }

                // Read stderr in background thread (cargo outputs build info here)
                if let Some(stderr) = child.stderr.take() {
                    let logs_clone = self.logs.clone();
                    let status_clone = self.status.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stderr);
                        for line in reader.lines() {
                            if let Ok(l) = line {
                                {
                                    let mut logs = logs_clone.lock().unwrap();
                                    logs.push_str("[BUILD] ");
                                    logs.push_str(&l);
                                    logs.push('\n');
                                }

                                // Detect compilation phase
                                let mut status = status_clone.lock().unwrap();
                                if l.contains("Compiling") || l.contains("Building") {
                                    status.is_compiling = true;
                                    status.message = l.clone();
                                } else if l.contains("Finished") || l.contains("Running") {
                                    status.is_compiling = false;
                                }
                            }
                        }
                    });
                }

                self.process = Some(child);
            }
            Err(e) => {
                let mut logs = self.logs.lock().unwrap();
                logs.push_str(&format!(
                    "❌ {}: {}\n",
                    self.text("Failed to start", "起動失敗"),
                    e
                ));
            }
        }
    }

    fn stop_training(&mut self) {
        if let Ok(path) = std::env::current_dir() {
            let signal_path = path.join("stop_signal");
            match std::fs::File::create(&signal_path) {
                Ok(_) => {
                    let mut logs = self.logs.lock().unwrap();
                    logs.push_str(&format!(
                        "\n🛑 {}\n",
                        self.text(
                            "Stop signal sent. Waiting for save...",
                            "停止シグナルを送信しました。保存待機中..."
                        )
                    ));
                }
                Err(e) => {
                    let mut logs = self.logs.lock().unwrap();
                    logs.push_str(&format!("\n❌ Error creating stop signal: {}\n", e));
                }
            }
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- 1. Log Management (Lock & Trim) ---
        let logs_to_display = {
            let mut logs_guard = self.logs.lock().unwrap();
            let len = logs_guard.len();
            if len > 100000 {
                let tail = logs_guard.split_off(len - 80000);
                *logs_guard = tail;
            }
            let display_limit = 5000;
            if logs_guard.len() > display_limit {
                logs_guard[logs_guard.len() - display_limit..].to_string()
            } else {
                logs_guard.clone()
            }
        };

        // --- 2. Process Monitoring ---
        if let Some(ref mut child) = self.process {
            match child.try_wait() {
                Ok(Some(_)) => {
                    self.process = None;
                    *self.is_running.lock().unwrap() = false;
                    let mut logs = self.logs.lock().unwrap();
                    logs.push_str(&format!(
                        "\n✅ {}\n",
                        self.text("Training finished", "トレーニング完了")
                    ));
                }
                Ok(None) => {}
                Err(e) => {
                    let mut logs = self.logs.lock().unwrap();
                    logs.push_str(&format!("\n❌ Error: {}\n", e));
                }
            }
        }

        // --- 3. UI Render ---
        egui::CentralPanel::default().show(ctx, |ui| {
            // Header
            ui.horizontal(|ui| {
                ui.heading("🚀 Bit-TTT Trainer");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.selectable_value(&mut self.language, Language::English, "🇺🇸 English");
                    ui.selectable_value(&mut self.language, Language::Japanese, "🇯🇵 日本語");
                    ui.separator();
                    ui.checkbox(&mut self.use_gpu, "⚡ GPU (CUDA)");
                });
            });
            ui.separator();

            // Dashboard
            // IMPORTANT: Copy values and drop lock BEFORE any button handlers
            // to avoid deadlock when start_training tries to acquire the same lock
            let (step, total_steps, loss, lr, is_compiling, message) = {
                let status = self.status.lock().unwrap();
                (
                    status.step,
                    status.total_steps,
                    status.loss,
                    status.lr,
                    status.is_compiling,
                    status.message.clone(),
                )
            }; // Lock is dropped here

            let progress = if total_steps > 0 {
                step as f32 / total_steps as f32
            } else {
                0.0
            };

            ui.heading(format!("📊 {}", self.text("Progress", "進捗状況")));

            // Show compilation indicator if building
            if is_compiling {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(
                        egui::RichText::new(self.text(
                            "⏳ Compiling (this may take a few minutes)...",
                            "⏳ コンパイル中（数分かかることがあります）...",
                        ))
                        .strong()
                        .color(egui::Color32::YELLOW),
                    );
                });
            }

            ui.add(
                egui::ProgressBar::new(progress)
                    .show_percentage()
                    .animate(true),
            );

            egui::Grid::new("metrics").striped(true).show(ui, |ui| {
                ui.label(self.text("Step:", "ステップ:"));
                ui.label(format!("{} / {}", step, total_steps));
                ui.label(self.text("Loss:", "損失 (Loss):"));
                ui.label(
                    egui::RichText::new(format!("{:.4}", loss))
                        .strong()
                        .color(egui::Color32::LIGHT_RED),
                );
                ui.label(self.text("LR:", "学習率:"));
                ui.label(format!("{:.7}", lr));
                ui.end_row();
            });
            ui.label(egui::RichText::new(&message).italics().weak());
            ui.separator();

            // Settings Section using CollapsingHeader or grouping
            // Disable settings while running
            ui.add_enabled_ui(self.process.is_none(), |ui| {
                ui.heading(format!("⚙️ {}", self.text("Configuration", "設定")));

                egui::Grid::new("settings_grid")
                    .striped(true)
                    .spacing([20.0, 8.0])
                    .show(ui, |ui| {
                        // Checkpoint
                        ui.label(format!(
                            "📂 {}",
                            self.text("Checkpoint:", "チェックポイント:")
                        ));
                        ui.horizontal(|ui| {
                            if let Some(path) = &self.checkpoint_path {
                                ui.monospace(path);
                                if ui
                                    .button("🗑")
                                    .on_hover_text(self.text("Clear", "クリア"))
                                    .clicked()
                                {
                                    self.checkpoint_path = None;
                                }
                            } else {
                                ui.label(
                                    egui::RichText::new(self.text("(New Run)", "(新規学習)"))
                                        .weak(),
                                );
                            }
                            if ui.button(self.text("Load...", "参照...")).clicked() {
                                // Use catch_unwind to safely handle any panics in rfd
                                let result = std::panic::catch_unwind(|| {
                                    let dialog = rfd::FileDialog::new()
                                        .add_filter("SafeTensors", &["safetensors"]);
                                    let dialog = if let Ok(cwd) = std::env::current_dir() {
                                        dialog.set_directory(&cwd)
                                    } else {
                                        dialog
                                    };
                                    dialog.pick_file()
                                });

                                if let Ok(Some(path_buf)) = result {
                                    self.checkpoint_path = Some(path_buf.display().to_string());
                                }
                            }
                        });
                        ui.end_row();

                        // Data
                        ui.label(format!("💾 {}", self.text("Data Path:", "データパス:")));
                        ui.text_edit_singleline(&mut self.data_path);
                        ui.end_row();

                        // LR
                        ui.label(format!(
                            "📉 {}",
                            self.text("Learning Rate:", "学習率 (LR):")
                        ));
                        ui.add(egui::Slider::new(&mut self.lr, 0.00001..=0.01).logarithmic(true));
                        ui.end_row();

                        // Min LR
                        ui.label(format!("End LR (Cosine):"));
                        ui.add(
                            egui::Slider::new(&mut self.min_lr, 0.000001..=0.001).logarithmic(true),
                        );
                        ui.end_row();

                        // Steps
                        ui.label(format!("🔄 {}", self.text("Total Steps:", "総ステップ数:")));
                        ui.add(egui::DragValue::new(&mut self.steps).speed(100));
                        ui.end_row();

                        // Warmup
                        ui.label(format!(
                            "🔥 {}",
                            self.text("Warmup Steps:", "ウォームアップ:")
                        ));
                        ui.add(egui::DragValue::new(&mut self.warmup_steps).speed(10));
                        ui.end_row();

                        // Save Interval
                        ui.label(format!("💾 {}", self.text("Save Interval:", "保存間隔:")));
                        ui.add(egui::DragValue::new(&mut self.save_interval).speed(100));
                        ui.end_row();
                    });
            });

            ui.separator();

            // Actions
            ui.horizontal(|ui| {
                if self.process.is_none() {
                    if ui
                        .button(
                            egui::RichText::new(format!(
                                "▶ {}",
                                self.text("START Training", "学習開始")
                            ))
                            .heading()
                            .color(egui::Color32::WHITE)
                            .background_color(egui::Color32::DARK_GREEN),
                        )
                        .clicked()
                    {
                        self.start_training();
                    }
                } else {
                    if ui
                        .button(
                            egui::RichText::new(format!(
                                "⏹ {}",
                                self.text("STOP & SAVE", "保存して停止")
                            ))
                            .heading()
                            .color(egui::Color32::WHITE)
                            .background_color(egui::Color32::DARK_RED),
                        )
                        .clicked()
                    {
                        self.stop_training();
                    }
                    ui.spinner();
                }

                if ui
                    .button(format!("🗑 {}", self.text("Clear Log", "ログ消去")))
                    .clicked()
                {
                    self.logs.lock().unwrap().clear();
                }
            });

            ui.separator();

            // Logs
            ui.collapsing(format!("📋 {}", self.text("Logs", "ログ")), |ui| {
                egui::ScrollArea::vertical()
                    .max_height(200.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        ui.code(logs_to_display.as_str());
                    });
            });

            // Request repaint while running or compiling
            if self.process.is_some() || is_compiling {
                ctx.request_repaint();
            }
        });
    }
}

// アプリ終了時（ウィンドウを閉じた時）のクリーンアップ処理
impl Drop for MyApp {
    fn drop(&mut self) {
        if let Some(mut child) = self.process.take() {
            let _ = child.kill();
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
