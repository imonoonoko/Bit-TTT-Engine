use eframe::egui;
use std::env;
use std::process::Command;

use crate::data::preprocess::{self, PreprocessArgs};
use crate::gui::i18n::{t, t_tooltip, Language};
use crate::state::ProjectState;
use crate::vocab::ModelType;
use glob::glob;
use std::thread;

pub fn show_data_prep(ui: &mut egui::Ui, project: &mut ProjectState, language: Language) {
    ui.heading(t(language, "step1_title"));
    ui.label(t(language, "step1_desc"));
    ui.add_space(10.0);

    ui.group(|ui| {
        ui.heading(t(language, "collect_raw"));
        ui.horizontal(|ui| {
            if ui.button(t(language, "open_raw_folder")).clicked() {
                let _ = Command::new("explorer")
                    .arg(project.path.join("raw"))
                    .spawn();
            }
            ui.label(t(language, "place_txt_here"));
        });
    });

    ui.add_space(10.0);

    ui.group(|ui| {
        ui.heading(t(language, "concat_corpus"));

        let is_concatenating = project.is_running && project.status_message.contains("Concatenating");

        if is_concatenating {
             ui.horizontal(|ui| {
                 ui.spinner();
                 ui.label(t(language, "processing")); // Reusing generic processing string or just hardcode for now
                 if ui.button("🛑 Cancel").clicked() {
                     project.cancel_concat();
                 }
             });
        } else {
             // Disable if other process is running
             if ui.add_enabled(!project.is_running, egui::Button::new(t(language, "concat_btn"))).clicked() {
                project.concat_txt_files();
             }
        }

        if project.has_corpus {
            ui.label(egui::RichText::new(t(language, "corpus_ready")).color(egui::Color32::GREEN));
        } else {
            ui.label(egui::RichText::new(t(language, "corpus_missing")).color(egui::Color32::RED));
        }
    });

    ui.add_space(10.0);

    ui.group(|ui| {
        ui.heading(t(language, "train_tokenizer"));

        ui.horizontal(|ui| {
            ui.label(t(language, "model_type"))
                .on_hover_text(t_tooltip(language, "model_type"));
            ui.radio_value(
                &mut project.config.model_type,
                ModelType::Unigram,
                t(language, "model_unigram"),
            );
            ui.radio_value(
                &mut project.config.model_type,
                ModelType::Bpe,
                t(language, "model_bpe"),
            );
        });

        ui.horizontal(|ui| {
            ui.label(t(language, "vocab_size"));
            ui.add(egui::DragValue::new(&mut project.config.vocab_size).clamp_range(100..=65535))
                .on_hover_text(t_tooltip(language, "vocab_size"));
        });

        ui.checkbox(&mut project.fast_vocab, "⚡ Fast Mode (Sample 100MB)");
        ui.add_space(5.0);

        if ui.button(t(language, "start_tokenizer")).clicked() {
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

            let model_type_str = match project.config.model_type {
                ModelType::Bpe => "bpe",
                ModelType::Unigram => "unigram",
            };

            let exe = env::current_exe().unwrap_or_default();
            let exe_str = exe.to_string_lossy().to_string();
            let mut cmd_args = vec![
                "vocab",
                "--input",
                &corpus_path,
                "--vocab-size",
                &vocab_str,
                "--output",
                &output_path,
                "--model-type",
                model_type_str,
            ];

            if project.fast_vocab {
                cmd_args.push("--limit-mb");
                cmd_args.push("100");
            }

            project.run_command(&exe_str, &cmd_args);
        }

        if project.has_tokenizer {
            ui.label(
                egui::RichText::new(t(language, "tokenizer_ready")).color(egui::Color32::GREEN),
            );
        }
    });
}

pub fn show_preprocessing(ui: &mut egui::Ui, project: &mut ProjectState, language: Language) {
    ui.heading(t(language, "step2_title"));
    ui.label(t(language, "step2_desc"));
    ui.add_space(10.0);

    if !project.has_corpus || !project.has_tokenizer {
        ui.colored_label(egui::Color32::RED, t(language, "step1_incomplete"));
    }

    ui.group(|ui| {
        ui.heading(t(language, "dataset_conversion"));

        // 1. Input Pattern (Glob)
        // 1. Input Pattern (Glob)
        ui.label(t(language, "input_pattern"));
        ui.horizontal(|ui| {
            // Check for changes
            let response = ui.add(egui::TextEdit::singleline(&mut project.config.input_pattern).desired_width(300.0));

            if ui.button(t(language, "open_folder")).clicked() {
                if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                    let path_str = folder.to_string_lossy().to_string();
                    project.config.input_pattern = format!("{}/*.jsonl", path_str);
                    // Force refresh cache
                    project.matched_file_count = None;
                }
            }
            if ui.button(t(language, "use_raw_folder")).clicked() {
                 // Use relative path for cleaner UI
                 project.config.input_pattern = format!("projects/{}/raw/*", project.config.name);
                 project.matched_file_count = None;
            }

            // Performance Logic: Run glob only on change or if cache is missing
            if response.changed() || project.matched_file_count.is_none() {
                 if !project.config.input_pattern.is_empty() {
                     match glob(&project.config.input_pattern) {
                         Ok(paths) => {
                             // Filter extensions to match backend logic
                             let valid_exts = ["json", "jsonl", "txt", "md"];
                             let count = paths.filter_map(Result::ok).filter(|p| {
                                 if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
                                     valid_exts.contains(&ext)
                                 } else {
                                     false
                                 }
                             }).count();
                             project.matched_file_count = Some(count);
                         },
                         Err(_) => {
                             project.matched_file_count = None;
                         }
                     }
                 } else {
                     project.matched_file_count = Some(0);
                 }
            }
        });

        // Preview: Matched files (Cached)
        if let Some(count) = project.matched_file_count {
             ui.small(format!("{} {}", t(language, "matched_files"), count));
        } else if !project.config.input_pattern.is_empty() {
             ui.small(egui::RichText::new("Invalid glob pattern").color(egui::Color32::RED));
        }

        ui.add_space(5.0);

        // 2. Template Editor
        ui.checkbox(&mut project.config.use_template, t(language, "enable_template"));

        if project.config.use_template {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.label(t(language, "preset"));
                    if ui.button(t(language, "load_alpaca")).clicked() {
                        project.config.template = "User: {{instruction}}\nAI: {{output}}".to_string();
                    }
                    if ui.button(t(language, "load_chatml")).clicked() {
                        project.config.template = "<|im_start|>user\n{{instruction}}<|im_end|>\n<|im_start|>assistant\n{{output}}<|im_end|>".to_string();
                    }
                });

                ui.add(
                    egui::TextEdit::multiline(&mut project.config.template)
                        .font(egui::TextStyle::Monospace)
                        .code_editor()
                        .desired_width(f32::INFINITY)
                        .desired_rows(4)
                        .hint_text(t(language, "template_placeholder"))
                );
            });
        }

        ui.add_space(5.0);

        // 3. Start Button (Direct Integration)
        if ui.button(t(language, "start_conversion")).clicked() {
            let corpus_path = project.config.input_pattern.clone(); // Now explicit Glob

            // Legacy corpus fallback? No, we enforce Glob now.

            let tokenizer_path = project
                .path
                .join("data/tokenizer.json")
                .to_string_lossy()
                .into_owned();
            let output_dir = project.path.join("data/").to_path_buf();

            // Construct Args
            let args = PreprocessArgs {
                input: corpus_path,
                tokenizer: tokenizer_path.into(),
                output_dir,
                template: if project.config.use_template && !project.config.template.is_empty() {
                    Some(project.config.template.clone())
                } else {
                    None
                },
                list_key: None, // Can add UI for this later if needed
                val_ratio: 0.01,
                batch_size: 10000,
            };

            project.is_running = true;
            project.status_message = "Running Preprocessing...".to_string();
            project.log("🚀 Starting Universal Preprocessing (Direct Thread)...");

            let tx = project.log_tx.clone();

            // Clone args for thread
            thread::spawn(move || {
                match preprocess::run(args) {
                    Ok(_) => {
                        tx.send("✅ Processing Complete!".to_string()).unwrap();
                    }
                    Err(e) => {
                        tx.send(format!("❌ Error: {}", e)).unwrap();
                    }
                }
                // Send completion signal to reset UI state
                let _ = tx.send("<<PREPROCESS_DONE>>".to_string());
            });
        }

        if project.has_dataset {
            ui.label(egui::RichText::new(t(language, "dataset_ready")).color(egui::Color32::GREEN));
        }
    });
}
