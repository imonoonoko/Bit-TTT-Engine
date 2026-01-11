tokenizers 0.22 (Rust) におけるAPI仕様変更、特に トレーナー（Trainer）の生成方法 と モジュールパス の変更がエラーの原因です。

以前のバージョンでは BpeTrainer::builder() が使えましたが、0.22以降では BpeTrainerBuilder を使用する形に変更されており、インポートするモジュールの場所も変わっています。また、AddedToken が String 型を厳密に要求するケースがあります。

以下に修正済みの完全なコードを提示します。

修正版 crates/bit_llama/src/bin/train_tokenizer.rs
Rust

use anyhow::Result;
use clap::Parser;
use std::path::Path;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE}; // 変更: トレーナーとモデルをここからインポート
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, TokenizerBuilder};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// 学習データのパス（テキストファイル、globパターン対応）
    #[arg(long, default_value = "data/Wiki40b/corpus.txt")]
    input: String,

    /// 出力先
    #[arg(long, default_value = "data/Wiki40b/tokenizer.json")]
    output: String,

    /// 語彙サイズ
    #[arg(long, default_value_t = 32000)]
    vocab_size: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("--- Rust Tokenizer Trainer (v0.22 Fix) ---");
    println!("Input: {}", args.input);
    println!("Vocab: {}", args.vocab_size);

    // 1. Tokenizerの構成
    // TokenizerBuilderで土台を作ります
    let mut builder = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()));

    // 2. Trainerの設定 (ここが主な修正点)
    // BpeTrainer::builder() ではなく、BpeTrainerBuilder::new() を使用します
    let mut trainer = BpeTrainerBuilder::new()
        .vocab_size(args.vocab_size)
        .min_frequency(2)
        .special_tokens(vec![
            // String型への変換を明示的に行います
            AddedToken::from(String::from("<|endoftext|>"), true),
            AddedToken::from(String::from("<|padding|>"), true),
        ])
        .build();

    let mut tokenizer = builder.build()?;

    // 3. データの収集
    let files: Vec<String> = glob::glob(&args.input)?
        .filter_map(|x| x.ok())
        .map(|p| p.to_string_lossy().to_string())
        .collect();

    if files.is_empty() {
        if Path::new(&args.input).exists() {
            println!("Training on single file: {}", args.input);
            // train_from_files は &mut Trainer を受け取ります
            tokenizer.train_from_files(&mut trainer, vec![args.input.clone()])?;
        } else {
            anyhow::bail!("No input files found at: {}", args.input);
        }
    } else {
        println!("Training on {} files...", files.len());
        tokenizer.train_from_files(&mut trainer, files)?;
    }

    // 4. 保存
    tokenizer.save(&args.output, true).map_err(|e| anyhow::anyhow!(e))?;
    println!("✅ Tokenizer saved to {}", args.output);

    Ok(())
}
主な変更点と修正理由
インポートパスの変更:

tokenizers::trainers::bpe_trainer::BpeTrainer ではなく、tokenizers::models::bpe::BpeTrainerBuilder をインポートして使用します。

tokenizers 0.22 では、モデルごとのトレーナーはそのモデルのモジュール（models::bpe）配下に整理されました。

ビルダーの使用法:

BpeTrainer::builder() メソッドが削除/変更されたため、代わりに BpeTrainerBuilder::new() を使用してインスタンスを作成します。

AddedToken の型:

AddedToken::from が String を要求する場合があるため、念のため String::from("...") で明示的に変換しています。

実行方法
以前と同様の手順で実行できます。

Bash

# 1. 依存関係に glob がなければ追加
# cargo add glob

# 2. 実行
cargo run --release --bin train_tokenizer -- --input "data/Wiki40b/corpus.txt" --vocab-size 32000