use crate::gui::inference_session::InferenceEvent;
use crate::gui::{BitStudioApp, ChatMessage};
use eframe::egui;

pub fn render(app: &mut BitStudioApp, ui: &mut egui::Ui) {
    let session = &mut app.inference_session;

    // 1. Event Polling
    while let Ok(event) = session.event_rx.try_recv() {
        match event {
            InferenceEvent::Output(text) => {
                // Append to last message if Assistant, else create new
                if let Some(last) = app.chat_history.last_mut() {
                    if last.role == "Assistant" {
                        last.content.push_str(&text);
                    } else {
                        app.chat_history
                            .push(ChatMessage { role: "Assistant".to_string(), content: text });
                    }
                } else {
                    app.chat_history
                        .push(ChatMessage { role: "Assistant".to_string(), content: text });
                }
            }
            InferenceEvent::Ready => {
                // Could enable input, or log "Ready"
                // For now, we assume input is always enabled but maybe show indicator
            }
            InferenceEvent::Error(err) => {
                app.chat_history.push(ChatMessage {
                    role: "System".to_string(),
                    content: format!("Error: {}", err),
                });
            }
            InferenceEvent::Exit => {
                app.chat_history.push(ChatMessage {
                    role: "System".to_string(),
                    content: "Process Exited.".to_string(),
                });
                session.active_process = None; // Sync state
            }
        }
    }

    // 2. Header (Model Loading)
    ui.horizontal(|ui| {
        ui.heading("💬 Inference Playground");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if session.is_active() {
                // Settings Controls
                if let Some(proj) = &mut app.current_project {
                    ui.label("Temp:");
                    if ui
                        .add(
                            egui::DragValue::new(&mut proj.config.inference_temp)
                                .speed(0.01)
                                .clamp_range(0.0..=2.0),
                        )
                        .changed()
                    {
                        let cmd = format!("/temp {:.2}", proj.config.inference_temp);
                        session.send_message(&cmd);
                    }
                    ui.label("Len:");
                    if ui
                        .add(egui::DragValue::new(&mut proj.config.inference_max_tokens).speed(10))
                        .changed()
                    {
                        let cmd = format!("/len {}", proj.config.inference_max_tokens);
                        session.send_message(&cmd);
                    }
                }

                if ui.button("⏹ Unload Model").clicked() {
                    session.stop();
                    app.chat_history.push(ChatMessage {
                        role: "System".to_string(),
                        content: "Model Unloaded.".to_string(),
                    });
                }
                ui.label(egui::RichText::new("🟢 Active").color(egui::Color32::GREEN));
            } else {
                let is_training = app.current_project.as_ref().is_some_and(|p| p.is_running);
                ui.add_enabled_ui(!is_training, |ui| {
                    if ui.button("▶ Load Model").clicked() {
                        if let Some(proj) = &app.current_project {
                            let models_dir = proj.path.join("models");

                            // Scan for best model file
                            let mut candidates = Vec::new();
                            if let Ok(entries) = std::fs::read_dir(&models_dir) {
                                for entry in entries.flatten() {
                                    let path = entry.path();
                                    if path.is_file() {
                                        if let Some(ext) = path.extension() {
                                            if ext == "safetensors" || ext == "bitt" {
                                                if let Ok(metadata) = entry.metadata() {
                                                    if let Ok(modified) = metadata.modified() {
                                                        candidates.push((path, modified));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if candidates.is_empty() {
                                app.chat_history.push(ChatMessage {
                                    role: "System".to_string(),
                                    content: format!(
                                        "No compatible models (.safetensors/.bitt) found in {:?}",
                                        models_dir
                                    ),
                                });
                                return;
                            }

                            // Sort by newest first
                            candidates.sort_by(|a, b| b.1.cmp(&a.1));

                            let target_path = candidates.first().unwrap().0.clone();

                            let path_str = target_path.to_string_lossy().to_string();

                            match session.spawn(
                                &path_str,
                                proj.config.inference_temp,
                                proj.config.inference_max_tokens,
                            ) {
                                Ok(_) => {
                                    app.chat_history.push(ChatMessage {
                                        role: "System".to_string(),
                                        content: format!("Loading model from {}...", path_str),
                                    });
                                }
                                Err(e) => {
                                    app.chat_history.push(ChatMessage {
                                        role: "System".to_string(),
                                        content: format!("Failed to spawn: {}", e),
                                    });
                                }
                            }
                        } else {
                            app.chat_history.push(ChatMessage {
                                role: "System".to_string(),
                                content: "No project selected.".to_string(),
                            });
                        }
                    }
                })
                .response
                .on_disabled_hover_text(
                    "Training is active. Stop training to free VRAM for inference.",
                );
                ui.label(egui::RichText::new("⚪ Inactive").color(egui::Color32::GRAY));
            }
        });
    });
    ui.separator();

    // 3. Chat History Area
    // 3. Chat History Area
    egui::TopBottomPanel::bottom("input_area").resizable(false).min_height(60.0).show_inside(
        ui,
        |ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                if ui.button("Send").clicked() {
                    send_message(app);
                }
                if ui.button("🗑 Clear").clicked() {
                    app.chat_history.clear();
                    if app.inference_session.is_active() {
                        app.inference_session.send_message("/reset");
                    }
                }

                let input_id = ui.make_persistent_id("chat_input_box");
                let has_focus = ui.memory(|m| m.has_focus(input_id));

                // Pre-consume Ctrl+Enter if focused
                // This prevents TextEdit from seeing the key, avoiding newline insertion.
                let hit_send = has_focus
                    && ui.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Enter));

                if hit_send {
                    send_message(app);
                    // Keep focus
                    ui.memory_mut(|m| m.request_focus(input_id));
                }

                let _response = ui.add(
                    egui::TextEdit::multiline(&mut app.chat_input)
                        .id(input_id)
                        .desired_rows(2)
                        .desired_width(f32::INFINITY)
                        .hint_text("Type a message... (Ctrl+Enter to send)"),
                );
            });
        },
    );

    egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
        for msg in &app.chat_history {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("{}: ", msg.role)).strong());
                ui.label(&msg.content);
            });
            ui.add_space(5.0);
        }
    });
}

fn send_message(app: &mut BitStudioApp) {
    let text = app.chat_input.trim().to_string();
    if text.is_empty() {
        return;
    }

    // Add User Message
    app.chat_history.push(ChatMessage { role: "User".to_string(), content: text.clone() });

    // Send to process
    if app.inference_session.is_active() {
        app.inference_session.send_message(&text);
        // Add placeholder for assistant?
        // app.chat_history.push(ChatMessage { role: "Assistant".to_string(), content: String::new() });
    } else {
        app.chat_history.push(ChatMessage {
            role: "System".to_string(),
            content: "Model not loaded.".to_string(),
        });
    }

    app.chat_input.clear();
}
