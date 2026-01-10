use eframe::egui;
use std::process::Command;

use crate::gui::AppTab;
use crate::gui::BitStudioApp;

pub fn render_workspace(app: &mut BitStudioApp, ui: &mut egui::Ui) {
    let project = app.current_project.as_mut().unwrap();

    egui::ScrollArea::vertical().show(ui, |ui| {
        match app.tab {
            AppTab::Home => { /* Handled by SidePanel */ }
            AppTab::DataPrep => {
                ui.heading("📝 Step 1: Data Preparation");
                ui.label("Import text files to create a training corpus.");
                ui.add_space(10.0);

                ui.group(|ui| {
                    ui.heading("1. Collect Raw Material");
                    ui.horizontal(|ui| {
                        if ui.button("📂 Open raw/ folder").clicked() {
                            let _ = Command::new("explorer")
                                .arg(project.path.join("raw"))
                                .spawn();
                        }
                        ui.label("← Place .txt files here");
                    });
                });

                ui.add_space(10.0);

                ui.group(|ui| {
                    ui.heading("2. Concatenate (Create Corpus)");
                    if ui.button("🔄 Concatenate to corpus.txt").clicked() {
                        project.concat_txt_files();
                    }

                    if project.has_corpus {
                        ui.label(
                            egui::RichText::new("✅ corpus.txt ready").color(egui::Color32::GREEN),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new("❌ Missing corpus.txt").color(egui::Color32::RED),
                        );
                    }
                });

                ui.add_space(10.0);

                ui.group(|ui| {
                    ui.heading("3. Train Tokenizer");
                    ui.horizontal(|ui| {
                        ui.label("Vocab Size:");
                        ui.add(
                            egui::DragValue::new(&mut project.config.vocab_size)
                                .clamp_range(100..=65535),
                        );
                    });
                    ui.add_space(5.0);

                    if ui.button("▶ Start Tokenizer Training").clicked() {
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

                        project.run_command(
                            "cargo",
                            &[
                                "run",
                                "--release",
                                "--bin",
                                "train_tokenizer",
                                "--",
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
                            egui::RichText::new("✅ tokenizer.json ready")
                                .color(egui::Color32::GREEN),
                        );
                    }
                });
            }
            AppTab::Preprocessing => {
                ui.heading("🔢 Step 2: Preprocessing");
                ui.label("Convert text to binary ID sequence.");
                ui.add_space(10.0);

                if !project.has_corpus || !project.has_tokenizer {
                    ui.colored_label(egui::Color32::RED, "⚠️ Error: Step 1 not complete.");
                }

                ui.group(|ui| {
                    ui.heading("Dataset Conversion");
                    if ui.button("▶ Start Conversion (Parallel)").clicked() {
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

                        project.run_command(
                            "cargo",
                            &[
                                "run",
                                "--release",
                                "--bin",
                                "preprocess_parallel",
                                "--",
                                "--input",
                                &corpus_path,
                                "--tokenizer",
                                &tokenizer_path,
                                "--output-dir",
                                &prefix,
                            ],
                        );
                    }

                    if project.has_dataset {
                        ui.label(
                            egui::RichText::new("✅ train.u32 ready").color(egui::Color32::GREEN),
                        );
                    }
                });
            }
            AppTab::Training => {
                ui.heading("🧠 Step 3: Training");
                if !project.has_dataset {
                    ui.colored_label(egui::Color32::RED, "⚠️ Error: Step 2 not complete.");
                }

                ui.group(|ui| {
                    ui.heading("Current Config");
                    ui.label(format!(
                        "Dim: {} | Layers: {} | Context: {}",
                        project.config.model_dim, project.config.layers, project.config.context_len
                    ));
                    ui.add_space(5.0);
                    if ui.button("⚙ Change in Settings").clicked() {
                        app.tab = AppTab::Settings;
                    }
                });

                ui.add_space(10.0);

                ui.group(|ui| {
                    ui.heading("Controls");
                    ui.horizontal(|ui| {
                        if !project.is_running {
                            if ui.button("▶ START Training").clicked() {
                                let data_dir =
                                    project.path.join("data").to_string_lossy().into_owned();
                                let output_dir =
                                    project.path.join("models").to_string_lossy().into_owned();

                                // Args
                                let steps = project.config.steps.to_string();
                                let lr = project.config.lr.to_string();
                                let dim = project.config.model_dim.to_string();
                                let layers = project.config.layers.to_string();
                                let context = project.config.context_len.to_string();
                                let batch = project.config.batch_size.to_string();
                                let min_lr = project.config.min_lr.to_string();
                                let warmup = project.config.warmup_steps.to_string();
                                let save_int = project.config.save_interval.to_string();

                                project.run_command(
                                    "cargo",
                                    &[
                                        "run",
                                        "--release",
                                        "--features",
                                        "cuda",
                                        "--bin",
                                        "train_llama",
                                        "--",
                                        "--data",
                                        &data_dir, // Note: Script used --data file, implementation here uses dir? Check train_llama logic.
                                        // Actually launcher.rs lines 725 passed --data &data_dir.
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
                            if ui.button("⏹ STOP").clicked() {
                                project.stop_process();
                            }
                            ui.spinner();
                            ui.label(&project.status_message);
                        }
                    });
                });
            }
            AppTab::Settings => {
                ui.heading("⚙ Settings");
                ui.group(|ui| {
                    ui.heading("Architecture");
                    egui::Grid::new("arch_grid").striped(true).show(ui, |ui| {
                        ui.label("Model Dim:");
                        ui.add(
                            egui::DragValue::new(&mut project.config.model_dim)
                                .clamp_range(64..=4096)
                                .speed(64),
                        );
                        ui.end_row();
                        ui.label("Layers:");
                        ui.add(
                            egui::DragValue::new(&mut project.config.layers).clamp_range(1..=128),
                        );
                        ui.end_row();
                        ui.label("Context Len:");
                        ui.add(
                            egui::DragValue::new(&mut project.config.context_len)
                                .clamp_range(32..=8192)
                                .speed(32),
                        );
                        ui.end_row();
                        ui.end_row();
                        ui.label("Heads:");
                        ui.add(egui::DragValue::new(&mut project.config.n_heads));
                        ui.end_row();
                        ui.label("Vocab Size:");
                        ui.add(egui::DragValue::new(&mut project.config.vocab_size));
                        ui.end_row();
                    });

                    ui.add_space(10.0);
                    ui.heading("Hyperparameters");
                    egui::Grid::new("hyper_grid").striped(true).show(ui, |ui| {
                        ui.label("Batch Size:");
                        ui.add(
                            egui::DragValue::new(&mut project.config.batch_size)
                                .clamp_range(1..=512),
                        );
                        ui.end_row();
                        ui.label("Steps:");
                        ui.add(egui::DragValue::new(&mut project.config.steps));
                        ui.end_row();
                        ui.label("Learning Rate:");
                        ui.add(egui::DragValue::new(&mut project.config.lr).speed(0.0001));
                        ui.end_row();
                        ui.label("Min LR:");
                        ui.add(egui::DragValue::new(&mut project.config.min_lr).speed(0.0001));
                        ui.end_row();
                        ui.label("Warmup Steps:");
                        ui.add(egui::DragValue::new(&mut project.config.warmup_steps));
                        ui.end_row();
                        ui.label("Save Interval:");
                        ui.add(egui::DragValue::new(&mut project.config.save_interval));
                        ui.end_row();
                    });

                    ui.add_space(10.0);
                    let (vram_gb, msg, color) = project.config.estimate_vram_usage();
                    ui.colored_label(color, format!("VRAM Check: {:.2} GB - {}", vram_gb, msg));
                });

                ui.add_space(10.0);
                if ui.button("💾 Save Config").clicked() {
                    project.save_config();
                }
            }
        }
    });
}
