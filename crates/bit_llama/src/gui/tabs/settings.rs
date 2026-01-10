use eframe::egui;

use crate::gui::i18n::{t, t_tooltip, Language};
use crate::gui::presets::ModelPreset;
use crate::state::ProjectState;

pub fn show(
    ui: &mut egui::Ui,
    project: &mut ProjectState,
    language: Language,
    current_preset: &mut ModelPreset,
    _on_tab_change: impl FnMut(crate::gui::AppTab),
) {
    ui.heading(t(language, "settings_title"));

    // Preset Selector
    ui.group(|ui| {
        ui.heading(t(language, "preset"));
        ui.horizontal(|ui| {
            for preset in ModelPreset::all() {
                let is_selected = *preset == *current_preset;
                let text = preset.display_name(language == Language::Japanese);
                if ui.selectable_label(is_selected, text).clicked() {
                    *current_preset = *preset;
                    preset.apply(&mut project.config);
                }
            }
        });
        ui.label(format!("VRAM: {}", current_preset.vram_estimate()));
    });

    ui.add_space(10.0);

    ui.group(|ui| {
        ui.heading(t(language, "architecture"));
        egui::Grid::new("arch_grid").striped(true).show(ui, |ui| {
            ui.label(t(language, "model_dim"));
            if ui
                .add(
                    egui::DragValue::new(&mut project.config.model_dim)
                        .clamp_range(64..=4096)
                        .speed(64),
                )
                .on_hover_text(t_tooltip(language, "model_dim"))
                .changed()
            {
                *current_preset = ModelPreset::Custom;
            }
            ui.end_row();

            ui.label(t(language, "layers"));
            if ui
                .add(egui::DragValue::new(&mut project.config.layers).clamp_range(1..=128))
                .on_hover_text(t_tooltip(language, "layers"))
                .changed()
            {
                *current_preset = ModelPreset::Custom;
            }
            ui.end_row();

            ui.label(t(language, "context_len"));
            if ui
                .add(
                    egui::DragValue::new(&mut project.config.context_len)
                        .clamp_range(32..=8192)
                        .speed(32),
                )
                .on_hover_text(t_tooltip(language, "context_len"))
                .changed()
            {
                *current_preset = ModelPreset::Custom;
            }
            ui.end_row();

            ui.label(t(language, "heads"));
            if ui
                .add(egui::DragValue::new(&mut project.config.n_heads))
                .on_hover_text(t_tooltip(language, "heads"))
                .changed()
            {
                *current_preset = ModelPreset::Custom;
            }
            ui.end_row();

            ui.label(t(language, "vocab_size"));
            if ui
                .add(egui::DragValue::new(&mut project.config.vocab_size))
                .on_hover_text(t_tooltip(language, "vocab_size"))
                .changed()
            {
                *current_preset = ModelPreset::Custom;
            }
            ui.end_row();
        });

        ui.add_space(10.0);
        ui.heading(t(language, "hyperparameters"));
        egui::Grid::new("hyper_grid").striped(true).show(ui, |ui| {
            ui.label(t(language, "batch_size"));
            if ui
                .add(egui::DragValue::new(&mut project.config.batch_size).clamp_range(1..=512))
                .on_hover_text(t_tooltip(language, "batch_size"))
                .changed()
            {
                *current_preset = ModelPreset::Custom;
            }
            ui.end_row();

            ui.label(t(language, "steps"));
            if ui
                .add(egui::DragValue::new(&mut project.config.steps))
                .on_hover_text(t_tooltip(language, "steps"))
                .changed()
            {
                *current_preset = ModelPreset::Custom;
            }
            ui.end_row();

            ui.label(t(language, "learning_rate"));
            if ui
                .add(egui::DragValue::new(&mut project.config.lr).speed(0.0001))
                .on_hover_text(t_tooltip(language, "learning_rate"))
                .changed()
            {
                *current_preset = ModelPreset::Custom;
            }
            ui.end_row();

            ui.label(t(language, "min_lr"));
            if ui
                .add(egui::DragValue::new(&mut project.config.min_lr).speed(0.0001))
                .on_hover_text(t_tooltip(language, "min_lr"))
                .changed()
            {
                *current_preset = ModelPreset::Custom;
            }
            ui.end_row();

            ui.label(t(language, "warmup_steps"));
            if ui
                .add(egui::DragValue::new(&mut project.config.warmup_steps))
                .on_hover_text(t_tooltip(language, "warmup_steps"))
                .changed()
            {
                *current_preset = ModelPreset::Custom;
            }
            ui.end_row();

            ui.label(t(language, "save_interval"));
            if ui
                .add(egui::DragValue::new(&mut project.config.save_interval))
                .on_hover_text(t_tooltip(language, "save_interval"))
                .changed()
            {
                *current_preset = ModelPreset::Custom;
            }
            ui.end_row();
        });

        ui.add_space(10.0);
        let (vram_gb, msg, color) = project.config.estimate_vram_usage();
        ui.colored_label(
            color,
            format!("{} {:.2} GB - {}", t(language, "vram_check"), vram_gb, msg),
        );
    });

    ui.add_space(10.0);
    if ui.button(t(language, "save_config")).clicked() {
        project.save_config();
        // Since we are already in Settings, no need to navigate, but let's keep the hook
        // in case we want to give feedback like "Saved!" in the future.
    }
}
