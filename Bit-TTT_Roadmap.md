Bit-TTT エンジニアリングロードマップ

このドキュメントは、Bit-TTTプロジェクトのエンジニアリング工程と、各タスクの技術的な依存関係を視覚化したものです。

🗺️ ロードマップ・フローチャート

graph TD
    %% クラス定義
    classDef done fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724;
    classDef urgent fill:#f8d7da,stroke:#dc3545,stroke-width:4px,color:#721c24;
    classDef next fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#856404;
    classDef future fill:#e2e3e5,stroke:#adb5bd,stroke-width:1px,color:#383d41;

    %% ノード定義
    Start(("Phase A<br/>完了")):::done

    subgraph Phase_B0 ["🛑 Phase B-0: 緊急対応 (Structural Fix)"]
        B0["コマンド呼び出し修正<br/>cargo run → current_exe"]:::urgent
        noteB0["これがないと<br/>配布環境で動きません"]
        B0 -.- noteB0
    end

    subgraph Phase_B ["🚧 Phase B: 統合 (Integration)"]
        B1["ログ・ストリーミング<br/>state.rs → GUI Console"]:::next
        B2["設定の同期<br/>Config → CLI Args"]:::next
        B3["エラーハンドリング<br/>プロセス監視・停止"]:::next
    end

    subgraph Phase_C ["📊 Phase C: 可視化 (Visualization)"]
        C1["学習曲線グラフ<br/>Loss Parsing & Plot"]:::future
        C2["VRAMモニタリング<br/>Real usage check"]:::future
    end

    subgraph Phase_D ["⚡ Phase D: 完成 (Completion)"]
        D1["推論プレイグラウンド<br/>Chat UI"]:::future
        D2["配布用ビルド<br/>Release Build"]:::future
    end

    %% 依存関係の接続
    Start ==> B0
    B0 ==> B1
    B0 --> B2
    
    B1 ==> C1
    B1 --> B3
    
    C1 --> D1
    B3 --> D1
    
    B2 --> D2
    C2 --> D2

    %% スタイルの適用
    linkStyle 0,1,3 stroke-width:4px,stroke:#dc3545,fill:none;


📝 各フェーズの解説

🛑 Phase B-0: 緊急対応 (Structural Fix)

現在、アプリの実行が開発環境（cargo run）に依存している状態を修正します。

コマンド呼び出し修正: ui.rs 内での外部プロセス起動ロジックを修正し、current_exe() を使用して自身のバイナリをサブコマンドとして呼び出すように変更します。これにより、Python環境やRustツールチェーンがないユーザー環境でも動作するようになります。

🚧 Phase B: 統合 (Integration)

Rust（GUI）とPython（学習コア）の連携を深め、アプリとしての安定性を確保します。

ログ・ストリーミング: バックグラウンドで走る学習プロセスの標準出力をリアルタイムでGUI上のコンソールに表示します。

設定の同期: GUIで設定したパラメータ（学習率やステップ数など）を、確実にCLI引数として渡せるようにします。

エラーハンドリング: OOM（メモリ不足）や例外発生時に、GUIがフリーズせず適切にエラーメッセージを表示するようにします。

📊 Phase C: 可視化 (Visualization)

テキストログだけでなく、視覚的なフィードバックを強化します。

学習曲線グラフ: ログから Loss 値をパースし、リアルタイムで折れ線グラフを描画します。

VRAMモニタリング: メモリ使用量を監視し、クラッシュを未然に防ぐためのインジケータを表示します。

⚡ Phase D: 完成 (Completion)

オールインワンのLLM開発スタジオとして仕上げます。

推論プレイグラウンド: 学習中のモデルをロードし、その場でチャットテストができるタブを追加します。

配布用ビルド: 最終的なリリースビルドを作成し、配布可能なパッケージにします。