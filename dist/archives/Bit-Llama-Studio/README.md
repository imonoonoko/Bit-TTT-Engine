# Bit-Llama Studio

[![Featured on Orynth](https://orynth.dev/api/badge/bit-ttt-engine?theme=dark&style=default)](https://orynth.dev/projects/bit-ttt-engine)

An all-in-one GUI studio for training, fine-tuning, and running compact Large Language Models (LLMs) on your local machine.

## Features
- **Zero Dependencies**: Single `.exe` file. No Python or CUDA installation required (optional but recommended for speed).
- **GUI Interface**: Easy-to-use tabs for Data Prep, Training, and Chat verification.
- **Embedded Japanese Support**: Proper font rendering for Japanese text out of the box.
- **Visual Training**: Real-time loss graph and system monitoring.

## ⚠️ Important: Unique Architecture & Compatibility
This application uses a **custom lightweight AI architecture** distinct from standard Transformers or GGUF formats.
Therefore, **it can only run models created within Bit-Llama Studio**. Please note that existing `.pth` or `.gguf` files are not compatible.

## 🚧 Status & Roadmap
This project is currently in the **Prototype stage**.
- It is designed to verify the fundamental cycle of training and inference on consumer hardware.
- Future plans include scaling up to larger models (7B/70B class) and adding advanced features.

## Installation
1.  Download `bit_llama.exe`.
2.  Place it in a folder of your choice (e.g., `Bit-Llama-Studio`).
3.  Double-click to run.

> [!NOTE]
> When you launch the application for the first time, a `projects` folder will be automatically created in the same directory. This is where your training data, configurations, and trained models will be saved.

## Usage
1.  **Data Prep**: Download sample data (e.g., Wiki40b) or import your own text files.
2.  **Preprocessing**: Clean and tokenize your data.
3.  **Training**: Configure hyperparameters and start training. Watch the loss curve!
4.  **Chat**: Test your trained model immediately in the Inference tab.

## Troubleshooting
- If the app crashes on startup, check the `logs/bit_llama.log` file created in the same folder.
- Ensure you have write permissions in the folder where `.exe` resides.
