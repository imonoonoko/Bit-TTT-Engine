use crate::gui::{AppTab, BitStudioApp, ChatMessage};
use eframe::egui;

pub fn render(app: &mut BitStudioApp, ui: &mut egui::Ui) {
    // NOTE: Event polling is now centralized in gui/mod.rs update() to work across all tabs

    // Capture simple states to avoid using app.inference_session in closures where possible
    let is_active = app.inference_session.is_active();
    let is_dreaming = app.inference_session.is_dreaming;

    // ---------------------------------------------------------
    // 2. Chat Header
    // ---------------------------------------------------------
    ui.horizontal(|ui| {
        ui.heading("💬 Chat");

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if is_active {
                if is_dreaming {
                    ui.label(
                        egui::RichText::new("💤 Dreaming...")
                            .color(egui::Color32::from_rgb(150, 200, 255)),
                    );
                } else {
                    ui.label(egui::RichText::new("🟢 Active").color(egui::Color32::GREEN));
                }
            } else {
                ui.label(egui::RichText::new("⚪ Inactive").color(egui::Color32::GRAY));
                if ui.button("➡ Go to Model Lab").clicked() {
                    app.tab = AppTab::ModelLab;
                }
            }
        });
    });

    ui.separator();

    // ---------------------------------------------------------
    // 3. Chat Area
    // ---------------------------------------------------------
    egui::TopBottomPanel::bottom("input_area")
        .resizable(false)
        .min_height(60.0)
        .show_inside(ui, |ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                // Send Button Locked
                ui.add_enabled_ui(!is_dreaming && is_active, |ui| {
                    let clicked = ui.button("Send").clicked();

                    let input_id = ui.make_persistent_id("chat_input_box");
                    let has_focus = ui.memory(|m| m.has_focus(input_id));
                    let hit_send = has_focus
                        && ui.input_mut(|i| {
                            i.consume_key(egui::Modifiers::COMMAND, egui::Key::Enter)
                        });

                    // Handle Sending
                    if clicked || hit_send {
                        let text = app.chat_input.trim().to_string();
                        if !text.is_empty() {
                            app.chat_history.push(ChatMessage {
                                role: "User".to_string(),
                                content: text.clone(),
                            });
                            if app.inference_session.is_active() {
                                app.inference_session.send_message(&text);
                            }
                            app.chat_input.clear();
                        }

                        if hit_send {
                            ui.memory_mut(|m| m.request_focus(input_id));
                        }
                    }

                    // Input Box
                    ui.add_sized(
                        ui.available_size(),
                        egui::TextEdit::multiline(&mut app.chat_input)
                            .id(input_id)
                            .hint_text(if !is_active {
                                "Load a model in Model Lab to chat..."
                            } else if is_dreaming {
                                "Dreaming... (Input Locked)"
                            } else {
                                "Type a message... (Ctrl+Enter to send)"
                            }),
                    );
                });
            });
        });

    // Chat History (Fill remaining space)
    egui::CentralPanel::default().show_inside(ui, |ui| {
        egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for msg in &app.chat_history {
                    ui.horizontal(|ui| {
                        let (color, prefix) = match msg.role.as_str() {
                            "User" => (egui::Color32::LIGHT_BLUE, "👤 User:"),
                            "Assistant" => (egui::Color32::GREEN, "🤖 AI:"),
                            "System" => (egui::Color32::YELLOW, "⚠ Sys:"),
                            _ => (egui::Color32::WHITE, "Unknown:"),
                        };

                        ui.label(egui::RichText::new(prefix).color(color).strong());
                        ui.label(&msg.content);
                    });
                }
            });
    });
}
