# Verification Plan: GUI Enhancement

## 1. Observability Design
*   **Log Streaming**: Verify that `stdout`/`stderr` from `train_llama` appears in the GUI console pane without significant delay (>1s).
*   **Process State**: Verify that the GUI accurately reflects "Running" vs "Idle" states, even if the process crashes.

## 2. Quality Assurance (QA)
### Smoke Test Protocol
1.  **Project Creation**: Create a new project "GuiTest". Check `projects/GuiTest/project.json` creation.
2.  **Config Save**: Edit "Layers" to 2, Click "Save". Check `project.json` for `layers: 2`.
3.  **VRAM Check**: Increase "Model Dim" to 4096. Check if VRAM warnings turn RED.
4.  **Dry Run**: Start training with dummy data. Check if "Running" spinner appears and logs flow.
5.  **Stop**: Click "STOP". Check if process terminates gracefully and UI returns to "Idle".

## 3. Automated Tests
*   We cannot easily unit test the GUI rendering (w/o headless setup), so we rely on manual verification via the protocol above.
*   We CAN unit test `ProjectConfig` logic (VRAM math).
