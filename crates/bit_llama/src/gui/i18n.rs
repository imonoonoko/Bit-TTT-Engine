//! Internationalization (i18n) - Language support for GUI
//!
//! Provides EN/JA translations for all UI text.

/// Supported languages
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Language {
    #[default]
    English,
    Japanese,
}

impl Language {
    /// Toggle to the other language
    pub fn toggle(&self) -> Self {
        match self {
            Language::English => Language::Japanese,
            Language::Japanese => Language::English,
        }
    }

    /// Display name for the language
    pub fn display_name(&self) -> &str {
        match self {
            Language::English => "English",
            Language::Japanese => "日本語",
        }
    }
}

/// Translate a key to the current language
/// Falls back to the key itself if not found
pub fn t(lang: Language, key: &str) -> &'static str {
    match (lang, key) {
        // === App Title ===
        (Language::Japanese, "app_title") => "Bit-TTT Studio",
        (Language::English, "app_title") => "Bit-TTT Studio",

        // === Tabs ===
        (Language::Japanese, "tab_home") => "🏠 ホーム",
        (Language::English, "tab_home") => "🏠 Home",
        (Language::Japanese, "tab_data") => "📝 データ準備",
        (Language::English, "tab_data") => "📝 Data Prep",
        (Language::Japanese, "tab_preprocess") => "🔢 前処理",
        (Language::English, "tab_preprocess") => "🔢 Preprocess",
        (Language::Japanese, "tab_training") => "🧠 学習",
        (Language::English, "tab_training") => "🧠 Training",
        (Language::Japanese, "tab_settings") => "⚙ 設定",
        (Language::English, "tab_settings") => "⚙ Settings",

        // === Home ===
        (Language::Japanese, "new_project") => "新規プロジェクト",
        (Language::English, "new_project") => "New Project",
        (Language::Japanese, "project_name") => "プロジェクト名:",
        (Language::English, "project_name") => "Project Name:",
        (Language::Japanese, "create_btn") => "📁 作成",
        (Language::English, "create_btn") => "📁 Create",
        (Language::Japanese, "existing_projects") => "既存プロジェクト",
        (Language::English, "existing_projects") => "Existing Projects",
        (Language::Japanese, "no_projects") => "プロジェクトがありません",
        (Language::English, "no_projects") => "No projects found",

        // === Data Preparation ===
        (Language::Japanese, "step1_title") => "📝 ステップ 1: データ準備",
        (Language::English, "step1_title") => "📝 Step 1: Data Preparation",
        (Language::Japanese, "step1_desc") => {
            "テキストファイルをインポートして学習用コーパスを作成します。"
        }
        (Language::English, "step1_desc") => "Import text files to create a training corpus.",
        (Language::Japanese, "collect_raw") => "1. 素材を収集",
        (Language::English, "collect_raw") => "1. Collect Raw Material",
        (Language::Japanese, "open_raw_folder") => "📂 raw/ フォルダを開く",
        (Language::English, "open_raw_folder") => "📂 Open raw/ folder",
        (Language::Japanese, "place_txt_here") => "← .txt ファイルをここに配置",
        (Language::English, "place_txt_here") => "← Place .txt files here",
        (Language::Japanese, "concat_corpus") => "2. 結合 (コーパス作成)",
        (Language::English, "concat_corpus") => "2. Concatenate (Create Corpus)",
        (Language::Japanese, "concat_btn") => "🔄 corpus.txt に結合",
        (Language::English, "concat_btn") => "🔄 Concatenate to corpus.txt",
        (Language::Japanese, "corpus_ready") => "✅ corpus.txt 準備完了",
        (Language::English, "corpus_ready") => "✅ corpus.txt ready",
        (Language::Japanese, "corpus_missing") => "❌ corpus.txt がありません",
        (Language::English, "corpus_missing") => "❌ Missing corpus.txt",
        (Language::Japanese, "train_tokenizer") => "3. トークナイザー学習",
        (Language::English, "train_tokenizer") => "3. Train Tokenizer",
        (Language::Japanese, "vocab_size") => "語彙サイズ:",
        (Language::English, "vocab_size") => "Vocab Size:",
        (Language::Japanese, "start_tokenizer") => "▶ トークナイザー学習を開始",
        (Language::English, "start_tokenizer") => "▶ Start Tokenizer Training",
        (Language::Japanese, "tokenizer_ready") => "✅ tokenizer.json 準備完了",
        (Language::English, "tokenizer_ready") => "✅ tokenizer.json ready",

        // === Preprocessing ===
        (Language::Japanese, "step2_title") => "🔢 ステップ 2: 前処理",
        (Language::English, "step2_title") => "🔢 Step 2: Preprocessing",
        (Language::Japanese, "step2_desc") => "テキストをバイナリIDシーケンスに変換します。",
        (Language::English, "step2_desc") => "Convert text to binary ID sequence.",
        (Language::Japanese, "step1_incomplete") => "⚠️ エラー: ステップ 1 が完了していません",
        (Language::English, "step1_incomplete") => "⚠️ Error: Step 1 not complete.",
        (Language::Japanese, "dataset_conversion") => "データセット変換",
        (Language::English, "dataset_conversion") => "Dataset Conversion",
        (Language::Japanese, "start_conversion") => "▶ 変換を開始 (並列処理)",
        (Language::English, "start_conversion") => "▶ Start Conversion (Parallel)",
        (Language::Japanese, "dataset_ready") => "✅ train.u32 準備完了",
        (Language::English, "dataset_ready") => "✅ train.u32 ready",

        // === Training ===
        (Language::Japanese, "step3_title") => "🧠 ステップ 3: 学習",
        (Language::English, "step3_title") => "🧠 Step 3: Training",
        (Language::Japanese, "step2_incomplete") => "⚠️ エラー: ステップ 2 が完了していません",
        (Language::English, "step2_incomplete") => "⚠️ Error: Step 2 not complete.",
        (Language::Japanese, "current_config") => "現在の設定",
        (Language::English, "current_config") => "Current Config",
        (Language::Japanese, "change_in_settings") => "⚙ 設定で変更",
        (Language::English, "change_in_settings") => "⚙ Change in Settings",
        (Language::Japanese, "controls") => "コントロール",
        (Language::English, "controls") => "Controls",
        (Language::Japanese, "start_training") => "▶ 学習開始",
        (Language::English, "start_training") => "▶ START Training",
        (Language::Japanese, "stop_training") => "⏹ 停止",
        (Language::English, "stop_training") => "⏹ STOP",
        (Language::Japanese, "training_progress") => "📊 学習進捗",
        (Language::English, "training_progress") => "📊 Training Progress",
        (Language::Japanese, "no_training_data") => {
            "学習データがありません。学習を開始するとLoss曲線が表示されます。"
        }
        (Language::English, "no_training_data") => {
            "No training data yet. Start training to see the loss curve."
        }
        (Language::Japanese, "clear_graph") => "🗑 グラフをクリア",
        (Language::English, "clear_graph") => "🗑 Clear Graph",

        // === Settings ===
        (Language::Japanese, "settings_title") => "⚙ 設定",
        (Language::English, "settings_title") => "⚙ Settings",
        (Language::Japanese, "architecture") => "アーキテクチャ",
        (Language::English, "architecture") => "Architecture",
        (Language::Japanese, "model_dim") => "モデル次元:",
        (Language::English, "model_dim") => "Model Dim:",
        (Language::Japanese, "layers") => "レイヤー数:",
        (Language::English, "layers") => "Layers:",
        (Language::Japanese, "context_len") => "コンテキスト長:",
        (Language::English, "context_len") => "Context Len:",
        (Language::Japanese, "heads") => "ヘッド数:",
        (Language::English, "heads") => "Heads:",
        (Language::Japanese, "hyperparameters") => "ハイパーパラメータ",
        (Language::English, "hyperparameters") => "Hyperparameters",
        (Language::Japanese, "batch_size") => "バッチサイズ:",
        (Language::English, "batch_size") => "Batch Size:",
        (Language::Japanese, "steps") => "ステップ数:",
        (Language::English, "steps") => "Steps:",
        (Language::Japanese, "learning_rate") => "学習率:",
        (Language::English, "learning_rate") => "Learning Rate:",
        (Language::Japanese, "min_lr") => "最小学習率:",
        (Language::English, "min_lr") => "Min LR:",
        (Language::Japanese, "warmup_steps") => "ウォームアップ:",
        (Language::English, "warmup_steps") => "Warmup Steps:",
        (Language::Japanese, "save_interval") => "保存間隔:",
        (Language::English, "save_interval") => "Save Interval:",
        (Language::Japanese, "save_config") => "💾 設定を保存",
        (Language::English, "save_config") => "💾 Save Config",

        // === Presets ===
        (Language::Japanese, "preset") => "プリセット:",
        (Language::English, "preset") => "Preset:",
        (Language::Japanese, "preset_tiny") => "🐣 Tiny (テスト用)",
        (Language::English, "preset_tiny") => "🐣 Tiny (Testing)",
        (Language::Japanese, "preset_small") => "🐥 Small (推奨)",
        (Language::English, "preset_small") => "🐥 Small (Recommended)",
        (Language::Japanese, "preset_medium") => "🦅 Medium (高性能GPU)",
        (Language::English, "preset_medium") => "🦅 Medium (High-end GPU)",
        (Language::Japanese, "preset_custom") => "⚙ Custom",
        (Language::English, "preset_custom") => "⚙ Custom",

        // === VRAM ===
        (Language::Japanese, "vram_check") => "VRAM 確認:",
        (Language::English, "vram_check") => "VRAM Check:",

        // === Fallback ===
        // Return empty string for unknown keys (safe fallback)
        _ => "",
    }
}

/// Translate tooltip text
pub fn t_tooltip(lang: Language, key: &str) -> &'static str {
    match (lang, key) {
        // === Architecture ===
        (Language::Japanese, "model_dim") => "隠れ層の次元数。大きいほど表現力↑、VRAM消費↑\n推奨: 256 (Small) / 512 (Medium)",
        (Language::English, "model_dim") => "Hidden layer dimension. Higher = more expressive, more VRAM.\nRecommended: 256 (Small) / 512 (Medium)",

        (Language::Japanese, "layers") => "Transformerブロックの数。大きいほど深いモデル。\n推奨: 8 (Small) / 12 (Medium)",
        (Language::English, "layers") => "Number of transformer blocks. More = deeper model.\nRecommended: 8 (Small) / 12 (Medium)",

        (Language::Japanese, "context_len") => "一度に処理できるトークン数。\n長いほど文脈を理解できるがVRAM消費↑",
        (Language::English, "context_len") => "Maximum tokens processed at once.\nLonger = better context understanding, more VRAM.",

        (Language::Japanese, "heads") => "マルチヘッドアテンションのヘッド数。\n通常は hidden_dim / 64",
        (Language::English, "heads") => "Number of attention heads.\nUsually hidden_dim / 64.",

        (Language::Japanese, "vocab_size") => "トークナイザーの語彙サイズ。\n推奨: 8192〜16384",
        (Language::English, "vocab_size") => "Tokenizer vocabulary size.\nRecommended: 8192-16384.",

        // === Hyperparameters ===
        (Language::Japanese, "batch_size") => "1回の更新で処理するサンプル数。\n大きいほど安定・高速だがVRAM消費↑",
        (Language::English, "batch_size") => "Samples per update. Larger = more stable/faster, more VRAM.",

        (Language::Japanese, "steps") => "学習の総ステップ数。\n1000〜10000 が一般的。",
        (Language::English, "steps") => "Total training steps.\nTypically 1000-10000.",

        (Language::Japanese, "learning_rate") => "学習率 (LR)。大きすぎると発散、小さすぎると遅い。\n推奨: 1e-4 〜 3e-4",
        (Language::English, "learning_rate") => "Learning rate. Too high = unstable, too low = slow.\nRecommended: 1e-4 to 3e-4.",

        (Language::Japanese, "min_lr") => "コサインスケジュールの最小学習率。\n推奨: 1e-5 〜 1e-6",
        (Language::English, "min_lr") => "Minimum LR for cosine schedule.\nRecommended: 1e-5 to 1e-6.",

        (Language::Japanese, "warmup_steps") => "学習率を徐々に上げるステップ数。\n推奨: 全ステップの 5-10%",
        (Language::English, "warmup_steps") => "Steps to gradually increase LR.\nRecommended: 5-10% of total steps.",

        (Language::Japanese, "save_interval") => "チェックポイントを保存する間隔 (ステップ)。\n推奨: 500",
        (Language::English, "save_interval") => "Checkpoint save interval (steps).\nRecommended: 500.",

        // === Fallback ===
        _ => "",
    }
}
