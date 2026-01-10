# Analysis Report: GUI Enhancement (Phase C)

## 1. Structural Analysis (As-Is)

Currently, the Bit-Llama Studio GUI is built using `eframe` (egui) and is monolithic in nature.
Its primary structure is defined in `crates/bit_llama/src/gui/mod.rs` and `ui.rs`.

### Core Components
*   **BitStudioApp (`gui/mod.rs`)**:
    *   Manages Global State (`current_project`, `available_projects`).
    *   Manages UI State (`tab`, `new_project_name`).
    *   Updates logic (polling process, handling messages).
    *   Renders the Main Layout (Sidebar + Top/Bottom Panels).
*   **ProjectConfig (`config.rs`)**:
    *   Defines the data model for project settings.
    *   Includes logic for VRAM estimation (`estimate_vram_usage`).
*   **ProjectState (`state.rs`)**:
    *   Manages runtime state of a specific project (process handle, logs).
    *   *Note: Needs verification of `state.rs` content.*

### Current Limitations
1.  **Monolithic UI Render**: Adding complex features (like detailed monitoring graphs or advanced dataset tools) in the current structure might bloat `update` loop.
2.  **Basic Process Management**: Polling `try_wait()` in the main UI thread works for now, but might freeze UI if synchronous operations occur.
3.  **Limited Feedback**: User logs are just a text blob. No structured event stream in GUI.

## 2. Intent Verification (To-Be)

We aim to enhance the GUI to support **Advanced Training Workflows** and **Better Observability**.

*   **Goal**: Make "Bit-Llama Studio" a complete IDE for model training.
*   **Key Features Needed**:
    *   **Real-time Monitoring**: Loss curves, VRAM usage (actual vs estimated).
    *   **Advanced Config**: Expose more hyperparameters (which we added in `ProjectConfig` but UI might not show all).
    *   **Better Safety**: The VRAM estimation is a good start, but need per-step validation.
    *   **Internationalization**: Japanese font support is already there (`ui.rs`/`mod.rs`), need to ensure all new UI elements use it.

## 3. Scope of Change
*   **Files**:
    *   `src/gui/mod.rs`: Main loop enhancements.
    *   `src/gui/ui.rs`: Component library.
    *   `src/state.rs`: Enhanced process management.
    *   `config.rs`: (Already updated, UI needs to match).
