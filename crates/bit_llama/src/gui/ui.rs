//! UI Rendering - Main Router (Refactored)

use eframe::egui;

use crate::gui::tabs;
use crate::gui::AppTab;
use crate::gui::BitStudioApp;

pub fn render_workspace(app: &mut BitStudioApp, ui: &mut egui::Ui) {
    let lang = app.language;
    let tab = app.tab;

    // We don't verify project existence here as it's guaranteed by caller (mod.rs),
    // but to be safe/rusty we unwrap inside arms.

    egui::ScrollArea::vertical().show(ui, |ui| {
        match tab {
            AppTab::Home => { /* Handled by SidePanel */ }
            AppTab::DataPrep => {
                if let Some(project) = app.current_project.as_mut() {
                    tabs::data::show_data_prep(ui, project, lang);
                }
            }
            AppTab::Preprocessing => {
                if let Some(project) = app.current_project.as_mut() {
                    tabs::data::show_preprocessing(ui, project, lang);
                }
            }
            AppTab::Training => {
                if let Some(project) = app.current_project.as_mut() {
                    tabs::training::show(ui, project, &mut app.training_graph, lang, |new_tab| {
                        app.tab = new_tab
                    });
                }
            }
            AppTab::Inference => super::tabs::inference::render(app, ui),
            AppTab::Settings => {
                if let Some(project) = app.current_project.as_mut() {
                    tabs::settings::show(ui, project, lang, &mut app.current_preset, |new_tab| {
                        app.tab = new_tab
                    });
                }
            }
        }
    });
}
