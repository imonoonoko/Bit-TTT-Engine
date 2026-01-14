# GUI Refactoring Plan

## 1. Objective
Refactor the monolithic `crates/bit_llama/src/gui/ui.rs` into modular tab-based components to improve maintainability and readability. The `render_workspace` function currently contains all logic for Data Preparation, Preprocessing, Training, and Settings, making it difficult to navigate.

## 2. Proposed Structure

Current:
```text
crates/bit_llama/src/gui/
├── mod.rs (App state & Sidebar)
├── ui.rs (All workspace rendering)
├── i18n.rs
├── presets.rs
└── graph.rs
```

New:
```text
crates/bit_llama/src/gui/
├── mod.rs
├── ui.rs (Main router only)
├── tabs/ (NEW)
│   ├── mod.rs
│   ├── data.rs (DataPrep & Preprocessing tabs)
│   ├── training.rs (Training tab & controls)
│   └── settings.rs (Settings & Presets)
├── i18n.rs
├── presets.rs
└── graph.rs
```

## 3. Implementation Steps

1.  **Create Directory**: `crates/bit_llama/src/gui/tabs/`
2.  **Extract Settings**: Move `AppTab::Settings` logic to `tabs/settings.rs`.
3.  **Extract Data**: Move `AppTab::DataPrep` and `AppTab::Preprocessing` logic to `tabs/data.rs`.
4.  **Extract Training**: Move `AppTab::Training` logic to `tabs/training.rs`.
5.  **Update `ui.rs`**: Replace huge match arm bodies with function calls like `tabs::settings::render(...)`.
6.  **Update `mod.rs`**: Publish the new `tabs` module.

## 4. Safety
- No logic changes, only code movement.
- `cargo check` after every file creation to ensure imports are correct.
- Verify GUI functionality manually after refactor.

