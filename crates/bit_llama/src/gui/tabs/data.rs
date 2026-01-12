use eframe::egui;
use std::env;
use std::process::Command;

use crate::gui::i18n::{t, t_tooltip, Language};
use crate::state::ProjectState;
use crate::vocab::ModelType;

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
        if ui.button(t(language, "concat_btn")).clicked() {
            project.concat_txt_files();
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
                    "--model-type",
                    model_type_str,
                ],
            );
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
        if ui.button(t(language, "start_conversion")).clicked() {
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
                    "--output-dir",
                    &prefix,
                ],
            );
        }

        if project.has_dataset {
            ui.label(egui::RichText::new(t(language, "dataset_ready")).color(egui::Color32::GREEN));
        }
    });
}
