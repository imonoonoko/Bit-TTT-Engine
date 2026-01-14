use crate::gui::{BitStudioApp, ChatMessage};
use eframe::egui;

pub fn render(app: &mut BitStudioApp, ui: &mut egui::Ui) {
    ui.heading("🔬 Model Lab");
    ui.separator();

    let is_active = app.inference_session.is_active();
    let is_dreaming = app.inference_session.is_dreaming;

    // ---------------------------------------------------------
    // 1. Model Control Section
    // ---------------------------------------------------------
    ui.group(|ui| {
        ui.heading("🧠 Model Control");

        if is_active {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("🟢 Active").color(egui::Color32::GREEN));

                // UNLOAD
                // Disable if dreaming to prevent crash
                ui.add_enabled_ui(!is_dreaming, |ui| {
                    if ui.button("⏹ Unload Model").clicked() {
                        app.inference_session.stop();
                        app.inference_session.is_dreaming = false;
                        // Log to console only, not chat
                        if let Some(proj) = &mut app.current_project {
                            proj.log("⏹ Model Unloaded.");
                        }
                    }
                });
            });

            // Settings (Temp/Len)
            ui.collapsing("⚙ Inference Parameters", |ui| {
                if let Some(proj) = &mut app.current_project {
                    ui.horizontal(|ui| {
                        ui.label("Temp:");
                        if ui
                            .add_enabled(
                                !is_dreaming,
                                egui::DragValue::new(&mut proj.config.inference_temp)
                                    .speed(0.01)
                                    .clamp_range(0.0..=2.0),
                            )
                            .changed()
                        {
                            let cmd = format!("/temp {:.2}", proj.config.inference_temp);
                            app.inference_session.send_message(&cmd);
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Len:");
                        if ui
                            .add_enabled(
                                !is_dreaming,
                                egui::DragValue::new(&mut proj.config.inference_max_tokens)
                                    .speed(10),
                            )
                            .changed()
                        {
                            let cmd = format!("/len {}", proj.config.inference_max_tokens);
                            app.inference_session.send_message(&cmd);
                        }
                    });
                }
            });
        } else {
            ui.label(egui::RichText::new("⚪ Inactive").color(egui::Color32::GRAY));
            let is_training = app.current_project.as_ref().is_some_and(|p| p.is_running);

            ui.add_enabled_ui(!is_training, |ui| {
                if ui.button("▶ Load Model (Auto-detect)").clicked() {
                    // Logic to find and spawn model
                    let mut spawn_args = None;

                    if let Some(proj) = &app.current_project {
                        let models_dir = proj.path.join("models");
                        if let Ok(entries) = std::fs::read_dir(&models_dir) {
                            for e in entries.flatten() {
                                let p = e.path();
                                if p.extension()
                                    .map_or(false, |x| x == "safetensors" || x == "bitt")
                                {
                                    spawn_args = Some((
                                        p.to_string_lossy().to_string(),
                                        proj.config.inference_temp,
                                        proj.config.inference_max_tokens,
                                    ));
                                    break;
                                }
                            }
                        }
                    }

                    if let Some((path, temp, tokens)) = spawn_args {
                        match app.inference_session.spawn(&path, temp, tokens) {
                            Ok(_) => {
                                // Log to console only, not chat
                                if let Some(proj) = &mut app.current_project {
                                    proj.log(&format!("▶ Loading {}...", path));
                                }
                            }
                            Err(e) => {
                                // Errors DO go to chat
                                app.chat_history.push(ChatMessage {
                                    role: "System".to_string(),
                                    content: format!("Failed: {}", e),
                                });
                            }
                        }
                    } else {
                        app.chat_history.push(ChatMessage {
                            role: "System".to_string(),
                            content: "No model found in project/models/.".to_string(),
                        });
                    }
                }
            });

            if is_training {
                ui.label("⚠ Training in progress. Finish training to load model.");
            }
        }
    });

    ui.add_space(10.0);

    // ---------------------------------------------------------
    // 2. Soul Management Section
    // ---------------------------------------------------------
    if is_active {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.heading("👻 Soul Management");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(format!("Level: {}", app.soul_level))
                            .strong()
                            .color(egui::Color32::GOLD),
                    );
                });
            });
            ui.separator();

            // Dreaming Status
            if is_dreaming {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(
                        egui::RichText::new("💤 Dreaming... (Training in Progress)")
                            .color(egui::Color32::from_rgb(150, 200, 255))
                            .strong(),
                    );
                });
                ui.label("Do not unload model or close application.");
            }

            ui.add_space(5.0);

            ui.horizontal(|ui| {
                // Sleep / Wake
                if is_dreaming {
                    if ui.button("☀ Wake Up (Save)").clicked() {
                        app.inference_session.send_message("/wake");
                        // Log to console only, not chat
                        if let Some(proj) = &mut app.current_project {
                            proj.log("☀ Requesting graceful wake up...");
                        }
                    }
                } else {
                    if ui.button("🌙 Sleep (Offline Learning)").clicked() {
                        app.inference_session.send_message("/sleep");
                        app.inference_session.is_dreaming = true;
                        // Log to console only, not chat
                        if let Some(proj) = &mut app.current_project {
                            proj.log("💤 Entering Sleep Mode...");
                        }
                    }
                }

                ui.separator();

                // Load / Save
                // Disable if dreaming
                ui.add_enabled_ui(!is_dreaming, |ui| {
                    if ui.button("📂 Load Soul").clicked() {
                        let exe_path = std::env::current_exe().unwrap_or_default();
                        let souls_dir = exe_path
                            .parent()
                            .unwrap_or(std::path::Path::new("."))
                            .join("souls");
                        if let Some(path) = rfd::FileDialog::new()
                            .set_directory(&souls_dir)
                            .add_filter("Soul", &["soul"])
                            .pick_file()
                        {
                            app.inference_session
                                .send_message(&format!("/load {}", path.display()));
                            app.current_soul_path = Some(path);
                        }
                    }
                    if ui.button("💾 Save Soul").clicked() {
                        let exe_path = std::env::current_exe().unwrap_or_default();
                        let souls_dir = exe_path
                            .parent()
                            .unwrap_or(std::path::Path::new("."))
                            .join("souls");
                        if let Some(path) = rfd::FileDialog::new()
                            .set_directory(&souls_dir)
                            .add_filter("Soul", &["soul"])
                            .save_file()
                        {
                            app.inference_session
                                .send_message(&format!("/save {}", path.display()));
                            app.current_soul_path = Some(path);
                        }
                    }
                });
            });

            ui.add_space(5.0);
            ui.checkbox(&mut app.autosave_enabled, "Auto-save Soul on Exit");

            if let Some(path) = &app.current_soul_path {
                ui.label(format!(
                    "Current Soul: {}",
                    path.file_name().unwrap_or_default().to_string_lossy()
                ));
            } else {
                ui.label("Current Soul: (New / Default)");
            }
        });
    } else {
        ui.group(|ui| {
            ui.heading("👻 Soul Management");
            ui.label("Load a model to manage Soul.");
        });
    }
}
