# Bit-Llama Studio User Guide

Bit-Llama Studio is an all-in-one tool for training and experimenting with **1-bit LLMs (BitNet b1.58)**.
This guide walks you through the entire process, from data preparation to training and inference.

---

## 🚀 Getting Started

1.  **Launch**: Double-click `Bit-TTT.exe` (or `start_gui.bat`) in the folder.
2.  **Demo Mode**: Use `start_demo.bat` to try it out with a pre-configured project structure.

---

## 📋 Workflow

Create your own AI in 4 simple steps:

### 1. Data Prep
Prepare the "textbooks" and "dictionary" for your AI.

1.  **📂 Open Raw Folder**:
    *   Click to open the folder where you should place your text files (`.txt`, `.jsonl`).
    *   Plain text files (books, articles) or JSONL datasets are supported.
2.  **🔗 Concatenate Corpus**:
    *   Merges all files into a single `corpus.txt`.
3.  **🔤 Train Tokenizer**:
    *   **Vocab Size**: Size of the dictionary. 8000 (Small) to 32000 (Standard).
    *   **Fast Mode**: Recommended. Uses only the first 100MB of data for speed.
    *   Click **Train Tokenizer** to generate `tokenizer.json`.

### 2. Preprocessing
Convert text data into a binary format (Token IDs) that the AI can read efficiently.

1.  **Template**:
    *   Leave unchecked for plain text completion (novels, etc.).
    *   Select **Alpaca** or **ChatML** if your data is in `input`/`output` JSONL format.
2.  **▶ Start Conversion**:
    *   Converts text to `train.u32`. This may take a few minutes depending on data size.

### 3. Training
Train the AI using your GPU.

1.  **Profile**:
    *   **Consumer (8GB VRAM)**: For standard gaming PCs (GTX 1070/3060, etc.).
    *   **Server (24GB+ VRAM)**: For high-end GPUs.
    *   Selecting a profile automatically sets optimal parameters (Dim, Layers).
2.  **▶ Start Training**:
    *   Begins the training loop.
    *   **Graph**: Watch the **Loss** curve go down.
        *   Loss > 5.0: The AI is still babbling random noises.
        *   Loss < 3.0: It starts organizing words into sentences.
        *   Loss < 2.0: Can generate coherent text.
3.  **🛑 STOP (Save)**:
    *   You can stop anytime. The model is saved automatically ("Graceful Stop").

### 4. Inference
Chat with your trained model.

1.  **Load Model**:
    *   Select a `.safetensors` file from the `models` folder.
    *   **Note**: The model MUST be loaded with the **same Tokenizer** used during training.
2.  **Chat**:
    *   Type your message and press Enter!

---

## ❓ Troubleshooting

### Q. The output is gibberish / Garbled text!
**A. The model is likely undertrained.**
*   If the AI replies with random words or characters (e.g., "Egg seven time entering..."), it hasn't learned the language yet.
*   **Solution**:
    1.  **Train Longer**: Increase `Steps`. Wait until Loss generates below **3.0**.
    2.  **Check Data**: If your dataset is tiny (<1MB), the model might memorize it or fail to generalize. Add more data.
    3.  **Mismatched Tokenizer**: If you re-trained the Tokenizer, **you MUST redo Preprocessing and Training**. Loading a model with a different tokenizer (e.g., Vocab 8000 vs 16000) will result in complete gibberish.

### Q. Training says "Cpu" / 0.00 MB VRAM
**A. GPU not detected.**
*   Ensure you have an NVIDIA GPU and installed the latest drivers.
*   Bit-Llama Studio currently requires CUDA.

### Q. "Stream did not contain valid UTF-8" Error
**A. Fixed in v0.1.0.**
*   We implemented auto-sanitization for data streams. This should no longer occur.

---
**Enjoy your 1-bit AI Journey!**
Bit-TTT Team
