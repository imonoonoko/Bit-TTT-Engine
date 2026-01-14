# Bit-Llama Studio User Guide (v0.2.0)

Bit-Llama Studio is an all-in-one tool that allows anyone to easily train and experiment with **1 bit LLM (BitNet b1.58)**.
This guide explains the steps from data preparation to training and inference (chat).

---

## 🚀 Getting Started

1.  **Launch**: Double-click `start_gui.bat` (or `Bit-TTT.exe`) in the folder.
2.  **Demo Mode**: Use `start_demo.bat` to verify installation with a demo project setup.
    *   **Note**: The demo model (Sample 10M) is initialized with random weights, so the output will be gibberish. This is normal behavior.

---

## 📋 Workflow

To create an AI, proceed through the following 4 steps from left to right.

### 1. Data Prep
Prepare "text data" and a "tokenizer (dictionary)" for the AI to learn.

1.  **📂 Open Raw Folder**:
    *   Click this button to open the folder where you should place your text files (`.txt`, `.jsonl`).
    *   Public domain books or Wiki dumps are recommended for starters.
2.  **🔗 Concatenate Corpus**:
    *   Combines scattered files into a single `corpus.txt`.
3.  **🔤 Train Tokenizer**:
    *   **Vocab Size**: Choose dictionary size. 8000 (Small) ~ 32000 (Standard).
    *   **Fast Mode**: Recommended. Uses only the first 100MB for fast training.
    *   Clicking the button generates `tokenizer.json`.

### 2. Preprocessing
Converts text data into a "numerical format (binary)" that is easy for the AI to process.

1.  **Template**:
    *   Uncheck for simple text completion (novels, etc.).
    *   Select **Alpaca** or **ChatML** if using dialogue data (`input`/`output` format).
2.  **▶ Start Conversion**:
    *   May take a few minutes. `train.u32` will be generated upon completion.

### 3. Training
Train the AI model using your GPU.

1.  **Profile**:
    *   **Consumer (8GB VRAM)**: For general PCs.
    *   **Server (24GB+ VRAM)**: For high-end GPUs.
    *   Selecting this automatically loads the optimal model size settings (Dim, Layers).
2.  **▶ Start Training**:
    *   Starts training.
    *   **Graph**: Watch the **Loss** go down in the central graph.
        *   Loss > 5.0: Cannot speak yet.
        *   Loss < 3.0: Starts learning grammar.
        *   Loss < 2.0: Generates fairly natural text.
3.  **🛑 STOP (Save)**:
    *   You can stop anytime. Data is automatically saved.
    *   Saved as `checkpoint_step_XXXX.safetensors`.

### 4. Inference (Chat)
Talk to your trained AI.

1.  **Load Model**:
    *   Select a `.safetensors` file from the `models` folder.
    *   **Note**: The **same Tokenizer used for training** must be present in the same folder.
2.  **Chat**:
    *   Type a message in the box below and press Enter!
    *   Adjust **Temperature/Top-P** to control creativity.

---

## ❓ Troubleshooting

### Q. Output is gibberish ("zazaza..." or random symbols)
**A. The model is undertrained.**
*   The **Demo Version (Sample 10M)** starts from a random state, so it cannot speak meaningful words initially.
*   If this happens with your model, **continue training until Loss drops below 3.0**.
*   Speaking like an alien is a "normal growth process" until it learns language rules.

### Q. Garbled Text / Tofu (□) appears
**A. Fixed in v0.2.0.**
*   Older versions had issues rendering some font characters on Windows.
*   The latest version automatically filters unrenderable characters, so you should see clean output.

### Q. Text overflows the screen
**A. Fixed in v0.2.0.**
*   The chat layout has been improved to automatically wrap long lines.

### Q. Training does not use GPU (0.00 MB / Cpu)
**A. Check your CUDA drivers.**
*   An NVIDIA GPU is required. Please install the latest drivers.
*   v0.2.0 includes a "Fallback Mode" that allows standard GPU usage even if custom kernels fail to compile.

---
**Happy Hacking!**
Bit-TTT Team
