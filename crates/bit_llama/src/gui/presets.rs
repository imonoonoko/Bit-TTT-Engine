//! Model Presets - Predefined configurations for different use cases

use crate::config::ProjectConfig;

/// Model preset categories
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ModelPreset {
    /// Tiny model for testing (~100MB VRAM)
    Tiny,
    /// Small model for learning (1-2GB VRAM)
    #[default]
    Small,
    /// Medium model for production (4-8GB VRAM)
    Medium,
    /// Custom configuration
    Custom,
}

impl ModelPreset {
    /// Apply preset values to config
    pub fn apply(&self, config: &mut ProjectConfig) {
        match self {
            ModelPreset::Tiny => {
                config.model_dim = 64;
                config.layers = 4;
                config.context_len = 64;
                config.n_heads = 4;
                config.batch_size = 8;
                config.steps = 500;
                config.lr = 5e-4;
                config.min_lr = 1e-5;
                config.warmup_steps = 50;
                config.save_interval = 500;
                config.accum_steps = 1;
            }
            ModelPreset::Small => {
                config.model_dim = 256;
                config.layers = 8;
                config.context_len = 128;
                config.n_heads = 8;
                config.batch_size = 16;
                config.steps = 5000;
                config.lr = 3e-4;
                config.min_lr = 1e-5;
                config.warmup_steps = 100;
                config.save_interval = 500;
                config.accum_steps = 4; // Effective Batch 64
            }
            ModelPreset::Medium => {
                config.model_dim = 512;
                config.layers = 12;
                config.context_len = 256;
                config.n_heads = 8;
                config.batch_size = 16;
                config.steps = 10000;
                config.lr = 2e-4;
                config.min_lr = 1e-6;
                config.warmup_steps = 200;
                config.save_interval = 500;
                config.accum_steps = 8; // Effective Batch 128
            }
            ModelPreset::Custom => {
                // Don't modify config for Custom
            }
        }
    }

    /// Get display name for the preset
    pub fn display_name(&self, is_japanese: bool) -> &'static str {
        match (self, is_japanese) {
            (ModelPreset::Tiny, true) => "🐣 Tiny (テスト用)",
            (ModelPreset::Tiny, false) => "🐣 Tiny (Testing)",
            (ModelPreset::Small, true) => "🐥 Small (推奨)",
            (ModelPreset::Small, false) => "🐥 Small (Recommended)",
            (ModelPreset::Medium, true) => "🦅 Medium (高性能GPU)",
            (ModelPreset::Medium, false) => "🦅 Medium (High-end GPU)",
            (ModelPreset::Custom, true) => "⚙ Custom",
            (ModelPreset::Custom, false) => "⚙ Custom",
        }
    }

    /// Get VRAM estimate for the preset
    pub fn vram_estimate(&self) -> &'static str {
        match self {
            ModelPreset::Tiny => "~100 MB",
            ModelPreset::Small => "~1-2 GB",
            ModelPreset::Medium => "~4-8 GB",
            ModelPreset::Custom => "Varies",
        }
    }

    /// All presets for iteration
    pub fn all() -> &'static [ModelPreset] {
        &[ModelPreset::Tiny, ModelPreset::Small, ModelPreset::Medium, ModelPreset::Custom]
    }
}
