# Quick Plan: GUI 文字化け修正 (GUI Font Fix)

## 1. Context (背景)
GUI上の日本語テキストが「□□□」のように文字化け (Tofu) している。
Windows環境においてシステムフォントのロードに失敗している、または適切なフォールバックが行われていない可能性が高い。

## 2. Risk Check (リスク確認)
*   **影響範囲**: `crates/bit_llama/src/gui/mod.rs` (フォントロード処理) のみ。
*   **リスク**: 低 (Low)。失敗してもデフォルトフォントに戻るだけ。
*   **プロセス**: **Lite Process** を適用。

## 3. Core Implementation (実装内容)

### A. 診断機能の追加 (Diagnosis)
*   `setup_custom_fonts` 関数内に `println!` または `info!` ログを追加し、どのフォントパスを試行し、どれが成功/失敗したかをコンソールに出力させる。
    *   現状はサイレントに失敗しているため、原因特定が困難。

### B. フォントパスの拡充 (Expand Paths)
*   Windowsのシステムフォントパス候補を追加する。
    *   `C:/Windows/Fonts/YuGothM.ttc` (Medium)
    *   `C:/Windows/Fonts/YuGothR.ttc` (Regular)
    *   `C:/Windows/Fonts/yugothb.ttc` (Lowercase check)
    *   `C:/Windows/Fonts/meiryo.ttc`

### C. 埋め込みフォントの導入 (Bundled Font - Recommended)
*   Google Fonts から **Noto Sans JP** (Regular) をダウンロードし、`assets/fonts/` に配置する。
*   `include_bytes!` マクロを使用してバイナリに埋め込むか、実行時にロードする。
*   **メリット**: OS環境に依存せず確実に表示できる。

## 4. Verification (検証手順)
1.  `cargo run` でGUIを起動。
2.  コンソールログで `[Font] Loaded: ...` を確認。
3.  GUI上で日本語が表示されているか目視確認。

## 5. Decision (決定事項)
まずは **(A) 診断** と **(B) パスの拡充** を即時実施する。
それでも解決しない場合、**(C) 埋め込み** を行う。
