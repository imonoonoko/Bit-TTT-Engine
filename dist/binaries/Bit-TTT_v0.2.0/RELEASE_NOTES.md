# Bit-Llama Studio v0.2.0 Release Notes

**Bit-Llama Studio v0.2.0** brings major improvements to performance, stability, and user experience. This release introduces GPU support for all users and fixes critical rendering issues for non-Latin languages.

## 🌟 Highlights (主な機能)

### ⚡ GPU Acceleration & Fallback Mode
- **English**: Now supports CUDA-based inference on NVIDIA GPUs. If the custom 1-bit kernel cannot be compiled or loaded on your system, it automatically falls back to a standard "GPU Dequantization Mode", ensuring you still get better-than-CPU performance (approx. 10x-20x faster).
- **Japanese**: NVIDIA GPU での高速推論に対応しました。もし専用の1-bitカーネルが動かない環境でも、自動的に「GPUフォールバックモード」に切り替わり、CPUよりも高速（約10〜20倍）に動作します。

### 🛠️ UI & UX Improvements
- **Font Rendering Fix**: Fixed an issue where Japanese characters (and other non-Latin glyphs) were rendered as "Tofu" (□) on Windows. The app now prioritizes system fonts like Meiryo.
- **Sanitized Output**: The chat interface now automatically filters out unrenderable control characters and ANSI escape codes (`[32m`), providing a clean visual experience even when the model generates "garbage" text during early training.
- **Smart Layout**: Chat messages now wrap correctly, preventing long text from overflowing the screen.
- **文字化け修正**: Windowsにおいて日本語が「□（豆腐）」になる問題を修正しました。メイリオ等のシステムフォントを優先します。
- **表示崩れ修正**: 長い文章が画面外にはみ出す問題を修正し、自動で折り返されるようにしました。
- **出力のクリーン化**: 学習不足のモデルが出力する謎の制御コードやゴミ文字を自動的にフィルタリングし、画面を汚さないようにしました。

### 📦 Easier Launchers
- Added `start_gui.bat` for one-click launch of the GUI.
- Fixed `start_demo.bat` to include necessary assets (tokenizer) for immediate testing.

---

## 📝 Changelog

- **[Feature]** Added `--features cuda` build with automatic fallback mechanism.
- **[Fix]** Implemented `Meiryo` / `Yu Gothic` / `MS Gothic` font fallback chain for Windows.
- **[Fix]** Implemented ANSI escape code stripping in Chat UI.
- **[Fix]** Implemented Control Character & Replacement Character sanitization.
- **[Fix]** Fixed `start_demo.bat` executable name and missing assets.
- **[Doc]** Updated User Guide (JA/EN) with troubleshooting sections.

## ⚠️ Notes

- The **Demo Model (Sample 10M)** included in the `start_demo.bat` is initialized with **Random Weights**. It will output meaningless text (gibberish). This is intended for verifying installation, not for actual chat. Please train your own model!
- **Demoモデル**は重みがランダムな状態です。意味のある言葉は話しませんが、正常動作です。ご自身でデータを用意して学習させてください！

---

**Full Changelog**: https://github.com/Humin/Bit-TTT/compare/v0.1.0...v0.2.0
