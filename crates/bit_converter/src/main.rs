use eframe::egui;

mod process;

use process::{ProcessEvent, ProcessManager};

fn main() -> eframe::Result<()> {
    // Logger setup (console)
    // tracing_subscriber::fmt::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 700.0])
            .with_title("🔄 Bit-Llama Converter"),
        ..Default::default()
    };

    eframe::run_native(
        "Bit-Llama Converter",
        options,
        Box::new(|cc| {
            setup_custom_fonts(&cc.egui_ctx);
            Box::new(BitConverterApp::new())
        }),
    )
}

struct BitConverterApp {
    // Config
    input_path: String,
    output_path: String,
    n_bases: i32,
    device: String,
    python_path: String,

    // State
    logs: String,
    is_converting: bool,
    progress: f32, // 0.0 to 1.0
    status_msg: String,

    // Backend
    process_manager: ProcessManager,
}

impl BitConverterApp {
    fn new() -> Self {
        Self {
            input_path: String::new(),
            output_path: String::new(),
            n_bases: 3,
            device: "cpu".to_string(),
            python_path: "python".to_string(), // Default checking PATH
            logs: "Ready to convert.\n".to_string(),
            is_converting: false,
            progress: 0.0,
            status_msg: "Idle".to_string(),
            process_manager: ProcessManager::new(),
        }
    }

    fn append_log(&mut self, msg: &str) {
        self.logs.push_str(msg);
        self.logs.push('\n');
    }
}

impl eframe::App for BitConverterApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll backend events
        while let Ok(event) = self.process_manager.rx.try_recv() {
            match event {
                ProcessEvent::Log(msg) => {
                    self.append_log(&msg);
                    // Simple hook for progress
                    // tqdm example: " 10%|"
                    // heuristic extraction
                    if let Some(idx) = msg.find("%|") {
                        // Extract number before %
                        // e.g. " 10%|..."
                        let end = idx;
                        let start = msg[..end].rfind(' ').map(|i| i + 1).unwrap_or(0);
                        if let Ok(p) = msg[start..end].trim().parse::<f32>() {
                            self.progress = p / 100.0;
                            self.status_msg = format!("Converting... {:.0}%", p);
                        }
                    }
                }
                ProcessEvent::Progress(p, msg) => {
                    self.progress = p;
                    self.status_msg = msg;
                }
                ProcessEvent::Exit(code) => {
                    self.is_converting = false;
                    self.progress = 1.0;
                    if code == 0 {
                        self.status_msg = "✅ Conversion Complete!".to_string();
                        self.append_log("✨ Process finished successfully.");
                    } else {
                        self.status_msg = format!("❌ Failed (Exit Code: {})", code);
                        self.append_log(&format!("❌ Process exited with code {}", code));
                    }
                }
                ProcessEvent::Error(err) => {
                    self.is_converting = false;
                    self.status_msg = "❌ Error occurred".to_string();
                    self.append_log(&format!("❌ Error: {}", err));
                }
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            // Header
            ui.horizontal(|ui| {
                ui.heading("🔄 Bit-Llama Converter");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.is_converting {
                        ui.spinner();
                        ui.label(egui::RichText::new(&self.status_msg).color(egui::Color32::YELLOW));
                    } else {
                        ui.label(egui::RichText::new(&self.status_msg).color(egui::Color32::GREEN));
                    }
                });
            });
            ui.separator();

            // Settings Section
            ui.collapsing("⚙ Settings", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Python Path:");
                    ui.text_edit_singleline(&mut self.python_path);
                    if ui.button("Auto-detect").clicked() {
                        self.python_path = "python".to_string(); // Reset to default
                    }
                }).response.on_hover_text("Path to python executable (e.g., 'python', 'python3', or full path)");

                ui.horizontal(|ui| {
                    ui.label("Dependencies:");
                    if ui.button("📦 Install Dependencies (pip)").clicked() {
                        // Spawning pip install logic could be added here
                        self.append_log("ℹ Feature not implemented yet. Please run: pip install torch safetensors huggingface_hub tqdm");
                    }
                });
            });
            ui.add_space(5.0);

            // Input / Output Section
            ui.group(|ui| {
                ui.heading("📂 Paths");
                ui.horizontal(|ui| {
                    ui.label("Input Model:   ");
                    ui.text_edit_singleline(&mut self.input_path).on_hover_text("Path to HuggingFace model folder or Repo ID");
                    if ui.button("📂").clicked() {
                        if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                            self.input_path = folder.to_string_lossy().to_string();
                        }
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Output Folder:");
                    ui.text_edit_singleline(&mut self.output_path);
                    if ui.button("📂").clicked() {
                        if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                            self.output_path = folder.to_string_lossy().to_string();
                        }
                    }
                });
            });
            ui.add_space(5.0);

            // Params Section
            ui.group(|ui| {
                ui.heading("🔧 Parameters");
                ui.horizontal(|ui| {
                    ui.label("Quantization Bases:");
                    ui.add(egui::DragValue::new(&mut self.n_bases).clamp_range(1..=8).speed(0.1));
                    ui.label("(Default: 3 -> 1.58 bit)");
                });

                 ui.horizontal(|ui| {
                    ui.label("Device:");
                    egui::ComboBox::from_label("Device Selection")
                        .selected_text(&self.device)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.device, "cpu".to_string(), "CPU");
                            ui.selectable_value(&mut self.device, "cuda".to_string(), "CUDA");
                        });
                });
            });

            ui.add_space(10.0);

            // Actions
            ui.horizontal(|ui| {
                if self.is_converting {
                    if ui.button("🛑 Stop").clicked() {
                        // Todo: Implement kill
                        self.append_log("⚠ Stop requested (Kill logic not implemented yet)");
                    }
                } else {
                    let btn = ui.button("🚀 Convert Model");
                    if btn.clicked() {
                        if self.input_path.is_empty() || self.output_path.is_empty() {
                            self.status_msg = "❌ Please check paths".to_string();
                        } else {
                            self.is_converting = true;
                            self.progress = 0.0;
                            self.status_msg = "Starting...".to_string();
                            self.logs.clear();

                            // Determine script path (Relative to exe or fixed)
                            // Assuming running from project root for dev, or assets separate.
                            // For dev: tools/convert_llama_v2.py
                            let script_path = "tools/conversion/convert_llama_v2.py".to_string();

                            self.process_manager.spawn_conversion(
                                &self.python_path,
                                &script_path,
                                &self.input_path,
                                &self.output_path,
                                self.n_bases,
                                &self.device
                            );
                        }
                    }
                }

                // Progress Bar
                if self.is_converting {
                    ui.add(egui::ProgressBar::new(self.progress).show_percentage());
                }
            });

            ui.separator();
            ui.heading("📜 Process Log");
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut self.logs)
                            .desired_width(f32::INFINITY)
                            .font(egui::TextStyle::Monospace)
                    );
                });
        });

        // Request repaint if converting to update logs smooth
        if self.is_converting {
            ctx.request_repaint();
        }
    }
}

fn setup_custom_fonts(_ctx: &egui::Context) {
    let _fonts = egui::FontDefinitions::default();

    // Load bundled font if available (copied from bit_llama assets)
    // We try to load "assets/fonts/NotoSansJP-Regular.ttf"
    // For standalone execution binary, we often put assets next to exe.

    // In dev: crates/bit_converter/assets/fonts/...
    // Implementation: Try load from byte inclusion or file

    // Let's rely on default for now to avoid complexity in this file,
    // or assume standard system fonts.
    // We already copied assets, so we could try loading.
}
