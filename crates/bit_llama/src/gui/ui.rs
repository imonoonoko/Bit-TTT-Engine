//! UI Rendering - Main Router (Refactored)

use eframe::egui;

use crate::gui::tabs;
use crate::gui::AppTab;
use crate::gui::BitStudioApp;

pub fn render_workspace(app: &mut BitStudioApp, ui: &mut egui::Ui) {
    let lang = app.language;
    let project = app.current_project.as_mut().unwrap();

    egui::ScrollArea::vertical().show(ui, |ui| {
        match app.tab {
            AppTab::Home => { /* Handled by SidePanel */ }
            AppTab::DataPrep => {
                tabs::data::show_data_prep(ui, project, lang);
            }
            AppTab::Preprocessing => {
                tabs::data::show_preprocessing(ui, project, lang);
            }
            AppTab::Training => {
                // Avoid double mutable borrow by splitting fields
                tabs::training::show(ui, project, &mut app.training_graph, lang, |new_tab| {
                    app.tab = new_tab
                });
            }
            AppTab::Settings => {
                tabs::settings::show(ui, project, lang, &mut app.current_preset, |new_tab| {
                    app.tab = new_tab
                });
            }
        }
    });
}
