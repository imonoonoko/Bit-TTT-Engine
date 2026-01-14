# Master Roadmap: Bit-TTT Engine

## ✅ Completed Phases
- **Phase 1-4**: Core Engine (Rust), BitLinear, TTT, GUI Prototype.
- **Phase 5**: Python Integration (`cortex_rust` PyO3 bindings).
- **Refactor V2**: Directory Structure Deep Clean (`workspace/`, `dist/`).

---

## 🚀 Phase 6: Deployment & Adoption (Deployment Phase) [Complete]
*(Goal: "Pip install and run")*
- [x] **Python Wheel**: Build `cortex_rust` wheel for Windows.
- [x] **Sample Weights**: Forge 10M sample model for testing.
- [x] **Hello Script**: `examples/hello_bit_llama.py` (Zero-config).

## 📢 Phase 7: Showcase (Visibility Phase) [Complete]
*(Goal: "Prove it works")*
- [x] **Rich README (v2)**: 5-minute Quickstart.
- [x] **Benchmarks**: Comparison table (vs Llama.cpp).
- [ ] **Demo Video**: (Pending user action).

---

## 🤝 Phase 8: Ecosystem & Community (Usability Phase)
*(Goal: "Enable creation")*

### 8.1 Web UI (Browser Chat)
- **Tool**: `tools/web_ui.py` (Gradio or Streamlit).
- **Features**:
  - Load model from `dist/assets` or `workspace/models`.
  - Chat interface with history.
  - Parameter sliders (Temperature, Top-P).

### 8.2 Training Guide
- **Doc**: `docs/TRAINING_GUIDE.md`.
- **Content**:
  - "How to prepare `corpus.txt`".
  - "How to run training loop".
  - "How to export for inference".

### 8.3 Advanced Benchmarking
- **Script**: `tools/benchmark_suite.py`.
- **Metrics**: Long-context stability, TTT overhead measurement.

---

## � Future Phases (Legacy Identity)
*Note: Merged from previous roadmap*
- **Phase 14**: Japanese Proficiency (Tokenizer extension).
- **Phase 15**: Scaling to 70B (Multi-GPU/CPU offload).
- **Phase 16**: WebAssembly (Wasm) support.
