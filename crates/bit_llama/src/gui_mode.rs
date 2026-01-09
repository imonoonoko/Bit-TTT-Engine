use crate::chat::{Message, Role};
use cortex_rust::Llama;
use eframe::egui;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;

// --- Persistence State ---
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct ChatApp {
    model_path: Option<PathBuf>,
    // Settings
    system_prompt: String,
    use_gpu: bool,
    temperature: f64,
    max_tokens: usize,

    // Runtime only (not saved)
    #[serde(skip)]
    llama: Option<Arc<Mutex<Llama>>>,
    #[serde(skip)]
    history: Vec<Message>,
    #[serde(skip)]
    input_text: String,
    #[serde(skip)]
    is_generating: bool,
    #[serde(skip)]
    rx: Option<mpsc::Receiver<(String, bool)>>,
    #[serde(skip)]
    status_msg: String,

    // Performance Metrics
    #[serde(skip)]
    start_time: Option<Instant>,
    #[serde(skip)]
    generated_tokens: usize,
    #[serde(skip)]
    current_tps: f64,
}

impl Default for ChatApp {
    fn default() -> Self {
        Self {
            model_path: None,
            system_prompt: "You are a helpful AI assistant.".to_owned(),
            use_gpu: true,
            temperature: 0.7,
            max_tokens: 500,
            llama: None,
            history: Vec::new(),
            input_text: String::new(),
            is_generating: false,
            rx: None,
            status_msg: "Please load a model.".to_owned(),
            start_time: None,
            generated_tokens: 0,
            current_tps: 0.0,
        }
    }
}

pub fn run() -> eframe::Result<()> {
    // Log file setup (simple redirect to file? might conflict with console CLI usage.
    // In mixed mode, we probably just want console logs to show up in terminal if launched from terminal).
    // keeping it simple.

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([800.0, 700.0])
            .with_title("Bit-TTT Llama Inference (Ore-BITT Edition)"),
        ..Default::default()
    };

    eframe::run_native(
        "bit_ttt_app",
        options,
        Box::new(|cc| {
            // Load from storage
            let mut app = ChatApp::default();

            if let Some(storage) = cc.storage {
                if let Some(json) = storage.get_string("bit_ttt_app") {
                    if let Ok(loaded) = serde_json::from_str::<ChatApp>(&json) {
                        app = loaded;
                    }
                }
            }

            Box::new(app)
        }),
    )
}

impl eframe::App for ChatApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if let Ok(json) = serde_json::to_string(self) {
            storage.set_string("bit_ttt_app", json);
        }
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 1. Poll Receiver
        let mut finished_generation = false;
        if let Some(rx) = &self.rx {
            while let Ok((token, finished)) = rx.try_recv() {
                // Determine implicit system prompt/history if newly loaded (handled in load_model)
                if self.history.is_empty() {
                    // Should have been init by load_model or send_message
                }

                if let Some(last_msg) = self.history.last_mut() {
                    if let Role::AI = last_msg.role {
                        last_msg.content.push_str(&token);
                    } else {
                        // New AI message
                        self.history.push(Message {
                            role: Role::AI,
                            content: token,
                        });
                    }
                } else {
                    self.history.push(Message {
                        role: Role::AI,
                        content: token,
                    });
                }

                // Update TPS
                self.generated_tokens += 1;
                if let Some(start) = self.start_time {
                    let elapsed = start.elapsed().as_secs_f64();
                    if elapsed > 0.0 {
                        self.current_tps = self.generated_tokens as f64 / elapsed;
                    }
                }

                if finished {
                    finished_generation = true;
                }

                // Repaint for smooth streaming
                ctx.request_repaint();
            }
        }

        if finished_generation {
            self.is_generating = false;
            self.status_msg = "Generation Complete".to_string();
            self.rx = None; // Detach
        }

        // 2. UI Layout
        egui::SidePanel::left("settings_panel").show(ctx, |ui| {
            ui.heading("Settings");
            ui.separator();

            // --- Model Select ---
            ui.label("Model File:");
            let btn_text = self
                .model_path
                .as_ref()
                .map(|p| p.file_name().unwrap().to_string_lossy())
                .unwrap_or("Select Model...".into());

            if ui.button(format!("📂 {}", btn_text)).clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Bit-TTT Models", &["bitt", "safetensors"])
                    .pick_file()
                {
                    self.model_path = Some(path);
                    self.load_model();
                }
            }

            ui.separator();
            ui.label("System Prompt:");
            ui.add(
                egui::TextEdit::multiline(&mut self.system_prompt)
                    .hint_text("Enter system prompt...")
                    .desired_rows(3),
            );

            ui.separator();
            ui.label(format!("Temperature: {:.2}", self.temperature));
            ui.add(egui::Slider::new(&mut self.temperature, 0.1..=2.0));

            ui.separator();
            if ui
                .checkbox(&mut self.use_gpu, "⚡ Use GPU (CUDA)")
                .changed()
            {
                if self.llama.is_some() {
                    self.status_msg = "⚠️ Reload model to apply".to_string();
                }
            }

            ui.label(format!("Max Tokens: {}", self.max_tokens));
            ui.add(egui::Slider::new(&mut self.max_tokens, 10..=2000));

            ui.separator();
            if ui.button("Stop Generation").clicked() {
                self.is_generating = false;
                // Note: The thread continues in background but specific channel check fails?
                // Currently we just stop reading rx. The thread might run until completion or error.
                // Correct logic involves `Arc<AtomicBool>` flag for cancellation.
                // For now, simple UI disconnect.
                self.rx = None;
                self.status_msg = "Generation Stopped (UI disconnected)".to_string();
            }

            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.label("Bit-TTT v0.1");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Header
            ui.horizontal(|ui| {
                ui.heading("Bit-Llama Chat");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(&self.status_msg);
                    if self.is_generating {
                        ui.spinner();
                    }
                    if self.current_tps > 0.0 {
                        ui.label(format!("Speed: {:.1} t/s", self.current_tps));
                    }
                });
            });
            ui.separator();

            // Chat History
            egui::ScrollArea::vertical()
                .stick_to_bottom(true) // auto scroll to bottom
                .show(ui, |ui: &mut egui::Ui| {
                    for msg in &self.history {
                        let (bg_color, fg_color, align) = match msg.role {
                            Role::User => (
                                egui::Color32::from_rgb(45, 45, 60),
                                egui::Color32::WHITE,
                                egui::Align::RIGHT,
                            ),
                            Role::AI => (
                                egui::Color32::from_rgb(0, 50, 0),
                                egui::Color32::LIGHT_GREEN,
                                egui::Align::LEFT,
                            ),
                            Role::System => (
                                egui::Color32::from_gray(30),
                                egui::Color32::GRAY,
                                egui::Align::Center,
                            ),
                        };

                        ui.with_layout(egui::Layout::top_down(align), |ui| {
                            let text = egui::RichText::new(&msg.content).color(fg_color);
                            egui::Frame::none()
                                .fill(bg_color)
                                .rounding(5.0)
                                .inner_margin(8.0)
                                .show(ui, |ui| {
                                    ui.label(text);
                                });
                        });
                        ui.add_space(5.0);
                    }
                });

            // Input Area
            ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                ui.add_space(5.0);
                ui.horizontal(|ui| {
                    let response = ui.add_sized(
                        [ui.available_width() - 80.0, 40.0],
                        egui::TextEdit::singleline(&mut self.input_text)
                            .hint_text("Type a message...")
                            .lock_focus(true),
                    );

                    if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        if !self.is_generating && self.llama.is_some() {
                            self.send_message();
                            // Refocus
                            response.request_focus();
                        }
                    }

                    if ui
                        .add_enabled(
                            !self.is_generating && self.llama.is_some(),
                            egui::Button::new("Send").min_size([60.0, 40.0].into()),
                        )
                        .clicked()
                    {
                        self.send_message();
                    }
                });
                ui.separator();
            });
        });
    }
}

impl ChatApp {
    /// モデルロード処理
    fn load_model(&mut self) {
        if let Some(path) = &self.model_path {
            self.status_msg = format!("Loading model from {:?}...", path);

            match Llama::load_auto(path) {
                Ok(llama) => {
                    self.llama = Some(Arc::new(Mutex::new(llama)));
                    self.status_msg = "Model Loaded Successfully!".to_string();

                    // Reset history with system prompt
                    self.history.clear();
                    self.history.push(Message {
                        role: Role::System,
                        content: format!("System: {}", self.system_prompt),
                    });
                }
                Err(e) => {
                    self.status_msg = format!("Error: {}", e);
                    // Add error details to history for debugging
                    self.history.push(Message {
                        role: Role::System,
                        content: format!("Failed to load model:\n{}", e),
                    });
                }
            }
        }
    }

    /// メッセージ送信処理
    fn send_message(&mut self) {
        let text = self.input_text.trim().to_string();
        if text.is_empty() {
            return;
        }

        self.history.push(Message {
            role: Role::User,
            content: text.clone(),
        });
        self.input_text.clear();
        self.is_generating = true;
        self.status_msg = "Generating...".to_string();
        self.start_time = Some(Instant::now());
        self.generated_tokens = 0;
        self.current_tps = 0.0;

        // Clone for thread
        let llama_arc = self.llama.as_ref().unwrap().clone();
        let prompt = self
            .history
            .iter()
            .map(|msg| {
                format!(
                    "{}: {}",
                    match msg.role {
                        Role::User => "User",
                        Role::AI => "AI",
                        Role::System => "System",
                    },
                    msg.content
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\nAI: ";
        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);

        let max_tokens = self.max_tokens;
        let temp = self.temperature;

        thread::spawn(move || {
            let mut wrapper = llama_arc.lock().unwrap();

            // Result handling
            let _ = wrapper.stream_completion(&prompt, max_tokens, temp, |token| {
                // Send token to GUI
                // We use ignore error if channel closed (receiver dropped)
                if tx.send((token.to_string(), false)).is_err() {
                    return Ok(false); // Stop generation
                }
                Ok(true) // Continue
            });

            // Finished signal
            let _ = tx.send(("".to_string(), true));
        });
    }
}
