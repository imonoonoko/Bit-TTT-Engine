//! UI Rendering - Workspace content rendering with i18n support

use eframe::egui;
use std::env;
use std::process::Command;

use crate::gui::i18n::{t, t_tooltip, Language};
use crate::gui::presets::ModelPreset;
use crate::gui::AppTab;
use crate::gui::BitStudioApp;

pub fn render_workspace(app: &mut BitStudioApp, ui: &mut egui::Ui) {
    let lang = app.language;
    let project = app.current_project.as_mut().unwrap();

    egui::ScrollArea::vertical().show(ui, |ui| {
        match app.tab {
            AppTab::Home => { /* Handled by SidePanel */ }
            AppTab::DataPrep => {
                ui.heading(t(lang, "step1_title"));
                ui.label(t(lang, "step1_desc"));
                ui.add_space(10.0);

                ui.group(|ui| {
                    ui.heading(t(lang, "collect_raw"));
                    ui.horizontal(|ui| {
                        if ui.button(t(lang, "open_raw_folder")).clicked() {
                            let _ = Command::new("explorer")
                                .arg(project.path.join("raw"))
                                .spawn();
                        }
                        ui.label(t(lang, "place_txt_here"));
                    });
                });

                ui.add_space(10.0);

                ui.group(|ui| {
                    ui.heading(t(lang, "concat_corpus"));
                    if ui.button(t(lang, "concat_btn")).clicked() {
                        project.concat_txt_files();
                    }

                    if project.has_corpus {
                        ui.label(
                            egui::RichText::new(t(lang, "corpus_ready"))
                                .color(egui::Color32::GREEN),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new(t(lang, "corpus_missing"))
                                .color(egui::Color32::RED),
                        );
                    }
                });

                ui.add_space(10.0);

                ui.group(|ui| {
                    ui.heading(t(lang, "train_tokenizer"));
                    ui.horizontal(|ui| {
                        ui.label(t(lang, "vocab_size"));
                        ui.add(
                            egui::DragValue::new(&mut project.config.vocab_size)
                                .clamp_range(100..=65535),
                        )
                        .on_hover_text(t_tooltip(lang, "vocab_size"));
                    });
                    ui.add_space(5.0);

                    if ui.button(t(lang, "start_tokenizer")).clicked() {
                        let corpus_path = project
                            .path
                            .join("data/corpus.txt")
                            .to_string_lossy()
                            .into_owned();
                        let vocab_str = project.config.vocab_size.to_string();
                        let output_path = project
                            .path
                            .join("data/tokenizer.json")
                            .to_string_lossy()
                            .into_owned();

                        let exe = env::current_exe().unwrap_or_default();
                        let exe_str = exe.to_string_lossy().to_string();
                        project.run_command(
                            &exe_str,
                            &[
                                "vocab",
                                "--input",
                                &corpus_path,
                                "--vocab-size",
                                &vocab_str,
                                "--output",
                                &output_path,
                            ],
                        );
                    }

                    if project.has_tokenizer {
                        ui.label(
                            egui::RichText::new(t(lang, "tokenizer_ready"))
                                .color(egui::Color32::GREEN),
                        );
                    }
                });
            }
            AppTab::Preprocessing => {
                ui.heading(t(lang, "step2_title"));
                ui.label(t(lang, "step2_desc"));
                ui.add_space(10.0);

                if !project.has_corpus || !project.has_tokenizer {
                    ui.colored_label(egui::Color32::RED, t(lang, "step1_incomplete"));
                }

                ui.group(|ui| {
                    ui.heading(t(lang, "dataset_conversion"));
                    if ui.button(t(lang, "start_conversion")).clicked() {
                        let corpus_path = project
                            .path
                            .join("data/corpus.txt")
                            .to_string_lossy()
                            .into_owned();
                        let tokenizer_path = project
                            .path
                            .join("data/tokenizer.json")
                            .to_string_lossy()
                            .into_owned();
                        let prefix = project.path.join("data/").to_string_lossy().into_owned();

                        let exe = env::current_exe().unwrap_or_default();
                        let exe_str = exe.to_string_lossy().to_string();
                        project.run_command(
                            &exe_str,
                            &[
                                "data",
                                "--input",
                                &corpus_path,
                                "--tokenizer",
                                &tokenizer_path,
                                "--output",
                                &prefix,
                            ],
                        );
                    }

                    if project.has_dataset {
                        ui.label(
                            egui::RichText::new(t(lang, "dataset_ready"))
                                .color(egui::Color32::GREEN),
                        );
                    }
                });
            }
            AppTab::Training => {
                ui.heading(t(lang, "step3_title"));
                if !project.has_dataset {
                    ui.colored_label(egui::Color32::RED, t(lang, "step2_incomplete"));
                }

                ui.group(|ui| {
                    ui.heading(t(lang, "current_config"));
                    ui.label(format!(
                        "Dim: {} | Layers: {} | Context: {}",
                        project.config.model_dim, project.config.layers, project.config.context_len
                    ));
                    ui.add_space(5.0);
                    if ui.button(t(lang, "change_in_settings")).clicked() {
                        app.tab = AppTab::Settings;
                    }
                });

                ui.add_space(10.0);

                ui.group(|ui| {
                    ui.heading(t(lang, "controls"));
                    ui.horizontal(|ui| {
                        if !project.is_running {
                            if ui.button(t(lang, "start_training")).clicked() {
                                let data_dir =
                                    project.path.join("data").to_string_lossy().into_owned();
                                let output_dir =
                                    project.path.join("models").to_string_lossy().into_owned();

                                let steps = project.config.steps.to_string();
                                let lr = project.config.lr.to_string();
                                let dim = project.config.model_dim.to_string();
                                let layers = project.config.layers.to_string();
                                let context = project.config.context_len.to_string();
                                let batch = project.config.batch_size.to_string();
                                let min_lr = project.config.min_lr.to_string();
                                let warmup = project.config.warmup_steps.to_string();
                                let save_int = project.config.save_interval.to_string();

                                let exe = env::current_exe().unwrap_or_default();
                                let exe_str = exe.to_string_lossy().to_string();
                                project.run_command(
                                    &exe_str,
                                    &[
                                        "train",
                                        "--data",
                                        &data_dir,
                                        "--output-dir",
                                        &output_dir,
                                        "--steps",
                                        &steps,
                                        "--lr",
                                        &lr,
                                        "--min-lr",
                                        &min_lr,
                                        "--warmup-steps",
                                        &warmup,
                                        "--dim",
                                        &dim,
                                        "--layers",
                                        &layers,
                                        "--context-len",
                                        &context,
                                        "--batch-size",
                                        &batch,
                                        "--save-interval",
                                        &save_int,
                                    ],
                                );
                            }
                        } else {
                            if ui.button(t(lang, "stop_training")).clicked() {
                                project.stop_process();
                            }
                            ui.spinner();
                            ui.label(&project.status_message);
                        }
                    });
                });

                ui.add_space(10.0);

                // Loss Visualization Graph
                ui.group(|ui| {
                    ui.heading(t(lang, "training_progress"));
                    if app.training_graph.data.is_empty() {
                        ui.label(t(lang, "no_training_data"));
                    } else {
                        ui.label(format!(
                            "Step: {} | Latest Loss: {:.4}",
                            app.training_graph.current_step,
                            app.training_graph.latest_loss().unwrap_or(0.0)
                        ));
                        app.training_graph.ui(ui);
                    }
                    ui.horizontal(|ui| {
                        if ui.button(t(lang, "clear_graph")).clicked() {
                            app.training_graph.clear();
                        }
                    });
                });
            }
            AppTab::Settings => {
                ui.heading(t(lang, "settings_title"));

                // Preset Selector
                ui.group(|ui| {
                    ui.heading(t(lang, "preset"));
                    ui.horizontal(|ui| {
                        for preset in ModelPreset::all() {
                            let is_selected = *preset == app.current_preset;
                            let text = preset.display_name(lang == Language::Japanese);
                            if ui.selectable_label(is_selected, text).clicked() {
                                app.current_preset = *preset;
                                preset.apply(&mut project.config);
                            }
                        }
                    });
                    ui.label(format!("VRAM: {}", app.current_preset.vram_estimate()));
                });

                ui.add_space(10.0);

                ui.group(|ui| {
                    ui.heading(t(lang, "architecture"));
                    egui::Grid::new("arch_grid").striped(true).show(ui, |ui| {
                        ui.label(t(lang, "model_dim"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut project.config.model_dim)
                                    .clamp_range(64..=4096)
                                    .speed(64),
                            )
                            .on_hover_text(t_tooltip(lang, "model_dim"))
                            .changed()
                        {
                            app.current_preset = ModelPreset::Custom;
                        }
                        ui.end_row();

                        ui.label(t(lang, "layers"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut project.config.layers)
                                    .clamp_range(1..=128),
                            )
                            .on_hover_text(t_tooltip(lang, "layers"))
                            .changed()
                        {
                            app.current_preset = ModelPreset::Custom;
                        }
                        ui.end_row();

                        ui.label(t(lang, "context_len"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut project.config.context_len)
                                    .clamp_range(32..=8192)
                                    .speed(32),
                            )
                            .on_hover_text(t_tooltip(lang, "context_len"))
                            .changed()
                        {
                            app.current_preset = ModelPreset::Custom;
                        }
                        ui.end_row();

                        ui.label(t(lang, "heads"));
                        if ui
                            .add(egui::DragValue::new(&mut project.config.n_heads))
                            .on_hover_text(t_tooltip(lang, "heads"))
                            .changed()
                        {
                            app.current_preset = ModelPreset::Custom;
                        }
                        ui.end_row();

                        ui.label(t(lang, "vocab_size"));
                        if ui
                            .add(egui::DragValue::new(&mut project.config.vocab_size))
                            .on_hover_text(t_tooltip(lang, "vocab_size"))
                            .changed()
                        {
                            app.current_preset = ModelPreset::Custom;
                        }
                        ui.end_row();
                    });

                    ui.add_space(10.0);
                    ui.heading(t(lang, "hyperparameters"));
                    egui::Grid::new("hyper_grid").striped(true).show(ui, |ui| {
                        ui.label(t(lang, "batch_size"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut project.config.batch_size)
                                    .clamp_range(1..=512),
                            )
                            .on_hover_text(t_tooltip(lang, "batch_size"))
                            .changed()
                        {
                            app.current_preset = ModelPreset::Custom;
                        }
                        ui.end_row();

                        ui.label(t(lang, "steps"));
                        if ui
                            .add(egui::DragValue::new(&mut project.config.steps))
                            .on_hover_text(t_tooltip(lang, "steps"))
                            .changed()
                        {
                            app.current_preset = ModelPreset::Custom;
                        }
                        ui.end_row();

                        ui.label(t(lang, "learning_rate"));
                        if ui
                            .add(egui::DragValue::new(&mut project.config.lr).speed(0.0001))
                            .on_hover_text(t_tooltip(lang, "learning_rate"))
                            .changed()
                        {
                            app.current_preset = ModelPreset::Custom;
                        }
                        ui.end_row();

                        ui.label(t(lang, "min_lr"));
                        if ui
                            .add(egui::DragValue::new(&mut project.config.min_lr).speed(0.0001))
                            .on_hover_text(t_tooltip(lang, "min_lr"))
                            .changed()
                        {
                            app.current_preset = ModelPreset::Custom;
                        }
                        ui.end_row();

                        ui.label(t(lang, "warmup_steps"));
                        if ui
                            .add(egui::DragValue::new(&mut project.config.warmup_steps))
                            .on_hover_text(t_tooltip(lang, "warmup_steps"))
                            .changed()
                        {
                            app.current_preset = ModelPreset::Custom;
                        }
                        ui.end_row();

                        ui.label(t(lang, "save_interval"));
                        if ui
                            .add(egui::DragValue::new(&mut project.config.save_interval))
                            .on_hover_text(t_tooltip(lang, "save_interval"))
                            .changed()
                        {
                            app.current_preset = ModelPreset::Custom;
                        }
                        ui.end_row();
                    });

                    ui.add_space(10.0);
                    let (vram_gb, msg, color) = project.config.estimate_vram_usage();
                    ui.colored_label(
                        color,
                        format!("{} {:.2} GB - {}", t(lang, "vram_check"), vram_gb, msg),
                    );
                });

                ui.add_space(10.0);
                if ui.button(t(lang, "save_config")).clicked() {
                    project.save_config();
                }
            }
        }
    });
}
