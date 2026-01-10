# Synergy & Value Design: GUI Observability

## 1. Efficiency Optimization
*   **Reuse `BitLoader` Metrics**: `BitLoader` already counts tokens. We can expose this via a temporary metric file (`metrics.json`) that the GUI polls, giving real-time progress beyond just "step count".
*   **Unified Config**: Since we successfully unified `ProjectConfig`, the GUI is now the *authoritative source* for training params. We can leverage this to create "Preset Configurations" (e.g., "Tiny Llama", "Standard 1B") that simply autofill the config struct.

## 2. Cross-Functional Synergy
*   **Safety Net Integration**: The GUI should respect the "Pre-Commit Hook" logic. Ideally, the GUI warns if the user tries to import a >99MB file into the `raw/` folder (though it's valid to have it there, just not commit it).
*   **Native Look & Feel**: We are using `eframe`. We can enable "Dark Mode" properly and match system fonts (already done for Japanese) to improve UX significantly.

## 3. Value Proposition
*   **"No-Code" Training**: The end user can go from raw text to a trained model without touching the terminal.
*   **Visual Confirmation**: Seeing the "VRAM estimation" turn Green/Red gives immediate confidence before starting a long job.
