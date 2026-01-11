# Quick Plan: Risk Remediation Phase 1

**Context**: `docs/RISK_ASSESSMENT.md` のリスクのうち、即時修正可能で効果の高いものをLite Processで対応する。

## 1. Scope
- **Target Files**:
    - `crates/bit_llama/src/gui/tabs/inference.rs`
    - `crates/bit_llama/src/gui/mod.rs` (Optional)
- **Objectives**:
    1. **VRAM Safety**: 学習中（`project.is_running == true`）は「Load Model」ボタンを無効化し、ツールチップで理由を表示する。
    2. **Model Loading Safety**: モデルファイルが見つからない場合、エンジンを起動せず、チャット履歴にエラーメッセージを表示する。

## 2. Risk Check
- **Low Risk**: 基本的にUIロジックの追加のみ。既存の学習機能や推論エンジンには影響しない。

## 3. Core Implementation
### VRAM Safety (`inference.rs`)
```rust
ui.add_enabled_ui(!app.current_project.as_ref().map_or(false, |p| p.is_running), |ui| {
    if ui.button("▶ Load Model").clicked() { ... }
}).on_disabled_hover_text("Training is running. Stop training to load model.");
```

### Model Loading Safety (`inference.rs`)
```rust
// Scan phase
if candidates.is_empty() {
    app.chat_history.push(ChatMessage { role: "System", content: "No model found in models/ directory." });
    return; // Early return, do NOT spawn
}
```

## 4. Verification
- [ ] 学習を開始し、Inferenceタブでボタンが無効化されていることを確認。
- [ ] 空のプロジェクト（モデルなし）で「Load Model」を押し、警告が出ることを確認。
