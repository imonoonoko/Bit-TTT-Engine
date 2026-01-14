pub mod graph;
pub mod i18n;
pub mod inference_session;
pub mod presets;
pub mod tabs;
pub mod ui;

use eframe::egui;
use std::fs;
use std::path::Path;

use crate::config::ProjectConfig;
use crate::gui::graph::TrainingGraph;
use crate::gui::i18n::Language;
use crate::gui::inference_session::InferenceSession;
use crate::gui::presets::ModelPreset;
use crate::state::ProjectState;

const PROJECTS_DIR: &str = "workspace/projects";

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum AppTab {
    Home,
    DataPrep,
    Preprocessing,
    Training,
    Inference,
    ModelLab,
    Settings,
}

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct BitStudioApp {
    // Global State
    pub tab: AppTab,
    pub runtime: tokio::runtime::Runtime,

    // i18n
    pub language: Language,

    // UI State
    pub new_project_name: String,
    pub current_project: Option<ProjectState>,
    pub current_preset: ModelPreset,

    // Inference State
    pub inference_session: InferenceSession,
    pub chat_history: Vec<ChatMessage>,
    pub chat_input: String,
    pub soul_level: u64,

    // Autosave State
    pub current_soul_path: Option<std::path::PathBuf>,
    pub autosave_enabled: bool,
    pub is_dreaming: bool,

    // Project Selection
    pub available_projects: Vec<String>,

    // Training Visualization
    pub training_graph: TrainingGraph,
    // System Monitor
}

impl Default for BitStudioApp {
    fn default() -> Self {
        // Initialize Tokio runtime
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create Tokio runtime");

        // Ensure projects dir exists
        let _ = fs::create_dir_all(PROJECTS_DIR);

        Self {
            tab: AppTab::Home,
            runtime,
            language: Language::default(),
            new_project_name: "MyModel".to_string(),
            current_project: None,
            current_preset: ModelPreset::default(),
            available_projects: Self::scan_projects(),
            training_graph: TrainingGraph::new(),
            inference_session: InferenceSession::new(),
            chat_history: Vec::new(),
            chat_input: String::new(),
            soul_level: 0,
            current_soul_path: None,
            autosave_enabled: true, // Default to true for "Life Awareness"
            is_dreaming: false,
        }
    }
}

impl BitStudioApp {
    pub fn scan_projects() -> Vec<String> {
        let mut projects = Vec::new();
        if let Ok(entries) = fs::read_dir(PROJECTS_DIR) {
            for entry in entries.flatten() {
                if let Ok(ft) = entry.file_type() {
                    if ft.is_dir() {
                        if let Ok(name) = entry.file_name().into_string() {
                            projects.push(name);
                        }
                    }
                }
            }
        }
        projects.sort();
        projects
    }

    pub fn create_project(&mut self) {
        let name = self.new_project_name.trim().to_string();
        if name.is_empty() {
            return;
        }

        let path = Path::new(PROJECTS_DIR).join(&name);
        if path.exists() {
            return;
        }

        // Create directory structure
        let _ = fs::create_dir_all(path.join("raw"));
        let _ = fs::create_dir_all(path.join("data"));
        let _ = fs::create_dir_all(path.join("models"));

        // Save default config
        let config = ProjectConfig {
            name: name.clone(),
            input_pattern: format!("projects/{}/raw/*", name),
            ..Default::default()
        };
        let config_json = serde_json::to_string_pretty(&config).unwrap();
        let _ = fs::write(path.join("project.json"), config_json);

        self.available_projects = Self::scan_projects();
        self.load_project(&name);
    }

    pub fn load_project(&mut self, name: &str) {
        let path = Path::new(PROJECTS_DIR).join(name);
        let config_path = path.join("project.json");

        let config = if config_path.exists() {
            let data = fs::read_to_string(&config_path).unwrap_or_default();
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            ProjectConfig::default()
        };

        self.current_project = Some(ProjectState::new(path, config));
        // Reset tab to DataPrep when loading? Or keep current?
        self.tab = AppTab::DataPrep;
    }

    fn poll_inference_events(&mut self) {
        use crate::gui::inference_session::InferenceEvent;
        let session = &mut self.inference_session;
        while let Ok(event) = session.event_rx.try_recv() {
            match event {
                InferenceEvent::Output(text) => {
                    // Mirror to Console Logs panel
                    if let Some(proj) = &mut self.current_project {
                        let cleaned = text.trim();
                        if !cleaned.is_empty() {
                            proj.log(cleaned);
                        }
                    }

                    // Detect Sleep State Transitions
                    if text.contains("Entering Sleep Mode") {
                        session.is_dreaming = true;
                    }
                    if text.contains("Sleep finished")
                        || text.contains("Waking up")
                        || text.contains("Dream interrupted")
                        || text.contains("Nightmare")
                    {
                        session.is_dreaming = false;
                    }

                    // Filter: only add actual AI responses to chat history
                    let is_system_log = text.contains("Portable Mode")
                        || text.contains("CWD set to")
                        || text.contains("bit_llama:")
                        || text.contains("INFO")
                        || text.contains("WARN")
                        || text.contains("Bit-Llama Inference")
                        || text.contains("Loading model from")
                        || text.contains("[Auto-Config]")
                        || text.contains("Model Loaded")
                        || text.contains("Memory loaded from")
                        || text.contains("Memory saved to")
                        || text.contains("Soul Level:")
                        || text.contains("Entering Sleep Mode")
                        || text.contains("Sleep finished")
                        || text.contains("Waking up")
                        || text.contains("[Generating...]")
                        || text.starts_with("---")
                        || text.starts_with(">")
                        || text.starts_with("📍")
                        || text.starts_with("✅")
                        || text.starts_with("📂")
                        || text.starts_with("💾")
                        || text.starts_with("🌟")
                        || text.starts_with("🔄")
                        || text.starts_with("🌡️")
                        || text.starts_with("📏")
                        || text.trim().is_empty()
                        || text.trim() == ".";

                    if !is_system_log {
                        if let Some(last) = self.chat_history.last_mut() {
                            if last.role == "Assistant" {
                                last.content.push_str(&text);
                            } else {
                                self.chat_history.push(ChatMessage {
                                    role: "Assistant".to_string(),
                                    content: text,
                                });
                            }
                        } else {
                            self.chat_history.push(ChatMessage {
                                role: "Assistant".to_string(),
                                content: text,
                            });
                        }
                    }
                }
                InferenceEvent::Ready => {}
                InferenceEvent::Error(err) => {
                    self.chat_history.push(ChatMessage {
                        role: "System".to_string(),
                        content: format!("Error: {}", err),
                    });
                    session.is_dreaming = false;
                }
                InferenceEvent::Exit => {
                    self.chat_history.push(ChatMessage {
                        role: "System".to_string(),
                        content: "Process Exited.".to_string(),
                    });
                    session.active_process = None;
                    session.is_dreaming = false;
                }
                InferenceEvent::SoulLevel(lvl) => {
                    self.soul_level = lvl;
                }
            }
        }
    }
}

impl eframe::App for BitStudioApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_pixels_per_point(1.2);

        // Poll System Monitor (1Hz internally)

        // Poll process status and update graph
        if let Some(project) = &mut self.current_project {
            // Drain background logs and extract training data
            let data_points = project.drain_logs_with_parse();
            for (step, loss) in data_points {
                self.training_graph.add_point(step, loss);
            }

            if project.is_running {
                if let Some(child) = &mut project.active_process {
                    if let Ok(Some(_status)) = child.try_wait() {
                        project.is_running = false;
                        project.task_type = crate::state::TaskType::None;
                        project.active_process = None;
                        project.log("Process finished.");
                        project.check_files();
                    }
                }
            }
        }

        // Poll Inference Events (CENTRAL)
        self.poll_inference_events();

        // Left Panel (Project Management)
        egui::SidePanel::left("project_panel")
            .resizable(true)
            .default_width(220.0)
            .show(ctx, |ui| {
                // Language Toggle at top
                ui.horizontal(|ui| {
                    if ui.button(self.language.display_name()).clicked() {
                        self.language = self.language.toggle();
                    }
                });
                ui.add_space(5.0);
                ui.heading(i18n::t(self.language, "existing_projects"));
                ui.separator();

                ui.collapsing(i18n::t(self.language, "new_project"), |ui| {
                    ui.horizontal(|ui| {
                        ui.label(i18n::t(self.language, "project_name"));
                        ui.text_edit_singleline(&mut self.new_project_name);
                    });
                    if ui.button(i18n::t(self.language, "create_btn")).clicked() {
                        self.create_project();
                    }
                });

                ui.separator();

                ui.label("Existing Projects:");
                let projects = self.available_projects.clone();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for proj in &projects {
                        let is_selected = self
                            .current_project
                            .as_ref()
                            .is_some_and(|p| p.config.name == *proj);
                        if ui
                            .selectable_label(is_selected, format!("📄 {}", proj))
                            .clicked()
                        {
                            self.load_project(proj);
                        }
                    }
                });

                ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
                    ui.add_space(10.0);
                    if self.current_project.is_some() && ui.button("❌ Close Project").clicked() {
                        self.current_project = None;
                        self.tab = AppTab::Home;
                        self.available_projects = Self::scan_projects();
                    }
                    ui.separator();
                });
            });

        // Main Content
        if let Some(project) = &mut self.current_project {
            // Log Panel
            egui::TopBottomPanel::bottom("log_panel")
                .resizable(true)
                .default_height(150.0)
                .show(ctx, |ui| {
                    ui.heading("📟 Console Logs");
                    egui::ScrollArea::vertical()
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            ui.add(
                                egui::TextEdit::multiline(&mut project.get_logs())
                                    .font(egui::TextStyle::Monospace)
                                    .desired_width(f32::INFINITY),
                            );
                        });
                });

            // Nav Panel
            egui::TopBottomPanel::top("nav_panel").show(ctx, |ui| {
                ui.add_space(5.0);
                ui.horizontal(|ui| {
                    ui.heading(format!("🚀 {}", project.config.name));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if project.is_running {
                            let btn_text = if project.task_type == crate::state::TaskType::Training
                            {
                                "🛑 STOP (Save)"
                            } else {
                                "🛑 STOP"
                            };

                            if ui.button(btn_text).clicked() {
                                if project.task_type == crate::state::TaskType::Training {
                                    project.request_stop();
                                } else {
                                    project.kill_process();
                                    project.cancel_concat();
                                }
                            }
                            ui.spinner();
                            ui.label(
                                egui::RichText::new("Running...").color(egui::Color32::YELLOW),
                            );
                        } else {
                            ui.label(egui::RichText::new("Idle").color(egui::Color32::GREEN));
                        }
                    });
                });
                ui.add_space(5.0);
                ui.separator();
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.tab, AppTab::DataPrep, "1. Data Prep");
                    ui.selectable_value(&mut self.tab, AppTab::Preprocessing, "2. Preprocessing");
                    ui.selectable_value(&mut self.tab, AppTab::Training, "3. Training");
                    ui.selectable_value(&mut self.tab, AppTab::Inference, "4. Chat");
                    ui.selectable_value(&mut self.tab, AppTab::ModelLab, "5. Model Lab");
                    ui.selectable_value(&mut self.tab, AppTab::Settings, "⚙ Settings");
                });
                ui.add_space(5.0);
            });

            // Workspace
            egui::CentralPanel::default().show(ctx, |ui| {
                crate::gui::ui::render_workspace(self, ui);
            });
        }
    }
}

pub fn run() -> Result<(), eframe::Error> {
    let mut options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };

    // Load Icon
    if let Ok(icon) = load_icon() {
        options.viewport = options.viewport.with_icon(icon);
    }

    eframe::run_native(
        "🚀 Bit-Llama Studio",
        options,
        Box::new(|cc| {
            setup_custom_fonts(&cc.egui_ctx);
            Box::new(BitStudioApp::default())
        }),
    )
}

fn load_icon() -> anyhow::Result<egui::IconData> {
    // Try bundling the icon first (Compile time)
    // We use include_bytes! to embed it
    let icon_bytes = include_bytes!("../../assets/icon.png");
    let image = image::load_from_memory(icon_bytes)?;
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();

    Ok(egui::IconData {
        rgba: rgba.into_raw(),
        width,
        height,
    })
}

fn setup_custom_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    tracing::info!("[GUI] Loading bundled font: NotoSansJP-Regular.ttf");

    // Load bundled font
    fonts.font_data.insert(
        "jp_font".to_owned(),
        egui::FontData::from_static(include_bytes!("../../assets/fonts/NotoSansJP-Regular.ttf")),
    );

    // Fallback settings (Prioritize Japanese font)
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "jp_font".to_owned());

    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .insert(0, "jp_font".to_owned());

    // Windows Fallback: Load System Fonts (Meiryo > Yu Gothic UI > MS Gothic)
    #[cfg(target_os = "windows")]
    {
        let candidates = [
            "C:\\Windows\\Fonts\\meiryo.ttc",   // Meiryo (Best for UI)
            "C:\\Windows\\Fonts\\yugothr.ttc",  // Yu Gothic Regular
            "C:\\Windows\\Fonts\\msgothic.ttc", // MS Gothic (Fallback)
        ];

        for path in candidates {
            if std::path::Path::new(path).exists() {
                if let Ok(data) = std::fs::read(path) {
                    tracing::info!("[GUI] Loading system font: {}", path);
                    fonts.font_data.insert(
                        "sys_font".to_owned(),
                        egui::FontData::from_owned(data).tweak(egui::FontTweak {
                            scale: 1.0,
                            ..Default::default()
                        }),
                    );

                    // Append system font as fallback
                    fonts
                        .families
                        .entry(egui::FontFamily::Proportional)
                        .or_default()
                        .push("sys_font".to_owned());

                    fonts
                        .families
                        .entry(egui::FontFamily::Monospace)
                        .or_default()
                        .push("sys_font".to_owned());

                    // Found one good font, stop looking
                    break;
                }
            }
        }
    }

    ctx.set_fonts(fonts);
    tracing::info!("[GUI] Fonts initialized.");
}
