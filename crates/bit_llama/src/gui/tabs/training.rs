use eframe::egui;
use std::env;
use std::path::Path;

use crate::gui::graph::TrainingGraph;
use crate::gui::i18n::{t, Language};
use crate::gui::AppTab;
use crate::state::ProjectState;

pub fn show(
    ui: &mut egui::Ui,
    project: &mut ProjectState,
    training_graph: &mut TrainingGraph,
    language: Language,
    mut on_tab_change: impl FnMut(AppTab),
) {
    ui.heading(t(language, "step3_title"));
    if !project.has_dataset {
        ui.colored_label(egui::Color32::RED, t(language, "step2_incomplete"));
    }

    ui.group(|ui| {
        ui.heading(t(language, "current_config"));
        ui.label(format!(
            "Dim: {} | Layers: {} | Context: {}",
            project.config.model_dim, project.config.layers, project.config.context_len
        ));
        ui.add_space(5.0);
        if ui.button(t(language, "change_in_settings")).clicked() {
            on_tab_change(AppTab::Settings);
        }
    });

    ui.add_space(10.0);

    ui.group(|ui| {
        ui.heading(t(language, "controls"));
        ui.horizontal(|ui| {
            if !project.is_running {
                if ui.button(t(language, "start_training")).clicked() {
                    let data_dir = project.path.join("data").to_string_lossy().into_owned();
                    let output_dir = project.path.join("models").to_string_lossy().into_owned();

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
                let stop_signal = Path::new("stop_signal").exists();
                let btn_text = if stop_signal {
                    "💀 Force Stop"
                } else {
                    "🛑 Stop Training"
                };
                let btn_color = if stop_signal {
                    egui::Color32::RED
                } else {
                    egui::Color32::YELLOW
                };

                if ui
                    .button(egui::RichText::new(btn_text).color(btn_color))
                    .clicked()
                {
                    if stop_signal {
                        project.kill_process();
                        // remove signal (handled by kill_process)
                    } else {
                        project.request_stop();
                    }
                }

                ui.spinner();
                if stop_signal {
                    ui.label("Stop Requested (Saving)...");
                } else {
                    ui.label(&project.status_message);
                }
            }
        });
    });

    ui.add_space(10.0);

    // Loss Visualization Graph
    ui.group(|ui| {
        ui.heading(t(language, "training_progress"));
        if training_graph.data.is_empty() {
            ui.label(t(language, "no_training_data"));
        } else {
            ui.label(format!(
                "Step: {} | Latest Loss: {:.4}",
                training_graph.current_step,
                training_graph.latest_loss().unwrap_or(0.0)
            ));
            training_graph.ui(ui);
        }
        ui.horizontal(|ui| {
            if ui.button(t(language, "clear_graph")).clicked() {
                training_graph.clear();
            }
        });
    });
}
