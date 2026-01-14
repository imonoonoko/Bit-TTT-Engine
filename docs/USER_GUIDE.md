# Bit-TTT Engine User Guide (v0.3.0)

This guide explains the basic usage of Bit-TTT Engine and the new features: "Model Lab" and "Sleep Mode".

---

## 🚀 Introduction

Bit-TTT Engine is **"A Digital Life Form growing in your PC"**.
It learns through conversation and consolidates memories by sleeping.

### Main Interface
When you launch the app, you will see the following tabs:

1.  **🔬 Model Lab**: A lab to manage the "Body" (Model) and "Soul" (Memory).
2.  **💬 Chat**: The main room for interacting with the AI.
3.  **⚙️ Settings**: Various parameter settings.

---

## 🔬 Model Lab

A newly established "AI Maintenance Room" in v0.3.0. You load the model here before starting a chat.

### 1. Load Model (Body)
1.  Click **Scan Models** to auto-detect models (`.safetensors`) in the `models/` folder.
2.  Select the model you want to load from the list and click **▶ Load Model**.
3.  Upon success, the status will turn "Active".

### 2. Manage Soul (Soul)
"Soul" refers to the memories (TTT state) learned by the AI.

*   **📂 Load Soul**: Loads a previously saved `.soul` file to restore memories.
*   **💾 Save Soul**: Saves the current memories to a file (Auto-save is recommended).
*   **Auto-save on Exit**: Check this to automatically save `.soul` when the app closes (**Recommended**).

---

## 💬 Chat & Sleep

This is where you interact with and nurture the AI.

### 1. Interact (Learn)
Just type in the text box and send. The AI learns from your words in real-time.
*   **Note**: This learning is "Short-term Memory". It disappears when you close the app.

### 2. Sleep Mode
To fix short-term memory into long-term memory (`.soul`), please let it "Sleep" periodically.

1.  Click the **🌙 Sleep (Offline Learning)** button.
2.  The AI will replay the conversation logs at high speed while "Dreaming...".
3.  Chat is locked during this time.
4.  When learning is finished or you click **☀ Wake Up (Save)**, memory is consolidated and the `.soul` file is updated.

> **💡 Hint**: After a long conversation, always make sure to Sleep and save memories.

---

## 🛠️ Project Management (Left Panel)

*   **New Project**: Creates a new AI (Project). Models and logs are independent for each project.
*   **Language**: You can switch languages (English / Japanese) with the button at the top right.

---

## ⚠️ Troubleshooting

### Q. Stuck at "Loading..."
*   Check the Console Log (Right Panel). If there are no errors, the initial load may take tens of seconds (especially if VRAM is low).

### Q. Stop button doesn't work
*   Fixed in v0.3.0. If it still doesn't stop, please terminate `bit_llama.exe` from Task Manager.

### Q. Memory is weird (Going crazy)
*   If it starts speaking gibberish, try loading an old `.soul` file in Model Lab to revert to "Yesterday's state".
