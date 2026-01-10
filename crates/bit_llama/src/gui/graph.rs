//! Training Graph - Loss visualization using egui_plot

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

/// Stores training metrics for visualization
pub struct TrainingGraph {
    /// Data points: (step, loss)
    pub data: Vec<[f64; 2]>,
    /// Current step counter
    pub current_step: u64,
}

impl Default for TrainingGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingGraph {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            current_step: 0,
        }
    }

    /// Add a new data point
    pub fn add_point(&mut self, step: f64, loss: f64) {
        self.data.push([step, loss]);
        self.current_step = step as u64;
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.data.clear();
        self.current_step = 0;
    }

    /// Render the graph
    pub fn ui(&self, ui: &mut egui::Ui) {
        let points: PlotPoints = self.data.iter().copied().collect();
        let line = Line::new(points)
            .color(egui::Color32::from_rgb(100, 200, 100))
            .name("Loss");

        Plot::new("training_loss_plot")
            .view_aspect(2.0)
            .x_axis_label("Step")
            .y_axis_label("Loss")
            .show(ui, |plot_ui| {
                plot_ui.line(line);
            });
    }

    /// Get the latest loss value
    pub fn latest_loss(&self) -> Option<f64> {
        self.data.last().map(|p| p[1])
    }
}
