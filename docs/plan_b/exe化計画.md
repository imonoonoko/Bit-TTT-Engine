Step 1. コンソール画面（黒い画面）を消す
デフォルトでは、GUIアプリを起動しても後ろにコマンドプロンプトが出てしまいます。 これを防ぐために、gui/src/main.rs（またはエントリポイントとなるファイル）の先頭に以下のおまじないを追加します。

Rust

// releaseビルドの時だけコンソールを消す（デバッグ時はログが見たいので消さない）
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // ...
}
Step 2. リリースビルドの実行
開発用（Debug）ではなく、最適化された配布用（Release）としてビルドします。ファイルサイズが小さくなり、動作が爆速になります。

Bash

cargo build --release
生成場所: target/release/launcher.exe（※プロジェクト名によります）

Step 3. 依存DLLの同梱 (最重要)
Rustのバイナリ自体は静的リンク（全部入り）されやすいですが、Deep Learning系ライブラリ（LibTorchなど）を使っている場合、.exe 単体では動きません。

もし tch-rs (LibTorch) をバックエンドに使っている場合：

症状: .exe をダブルクリックしても何も起きない、あるいは「xxx.dllが見つかりません」と出る。

対策: ビルド時に使用した libtorch の lib フォルダにある .dll ファイル群（torch_cpu.dll, c10.dll など）を、生成された .exe と同じフォルダにコピーする必要があります。

配布時のフォルダ構成イメージ:

Plaintext

MyAIApp/
├── launcher.exe      <-- これだけ渡しても動かない可能性大
├── torch_cpu.dll     <-- これらが必要
├── c10.dll
└── assets/           <-- 画像や設定ファイルなどがあれば
Step 4. アイコンの設定 (任意)
デフォルトの「無地のアイコン」だと味気ないので、.exe にアイコンを埋め込むのが一般的です。 これには winres というクレートを使います。

Cargo.toml の [build-dependencies] に winres を追加。

プロジェクトルートに build.rs を作成し、アイコン指定のコードを書く。

今後のロードマップへの影響
今回の「GUI実装計画」において、exe化を見据えて意識すべき点は一つだけです。

ファイルパスの扱い:

開発中は src/assets/icon.png などを相対パスで読み込めますが、exe化した後は「exeがある場所」を基準にパスを探すロジックが必要になることがあります。

対策: アセット類（画像やフォント）は、可能な限りバイナリの中に埋め込む（include_bytes! マクロを使う）と、配布時にファイル欠けのトラブルがなくなります。