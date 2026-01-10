use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

#[derive(Args, Debug, Clone)]
pub struct DataArgs {
    /// 入力コーパス (corpus.txt)
    #[arg(long)]
    pub input: PathBuf,

    /// トークナイザー設定 (tokenizer.json)
    #[arg(long)]
    pub tokenizer: PathBuf,

    /// 出力ディレクトリ
    #[arg(long)]
    pub output_dir: PathBuf,

    /// 検証データの割合 (0.01 = 1%)
    #[arg(long, default_value_t = 0.01)]
    pub val_ratio: f64,

    /// バッチサイズ (行数)。メモリに応じて調整。
    #[arg(long, default_value_t = 10_000)]
    pub batch_size: usize,
}

pub fn run(args: DataArgs) -> Result<()> {
    println!("🚀 Starting Parallel Preprocessing...");

    // 1. トークナイザー読み込み
    let tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Rayon内から参照するためにArc化
    let tokenizer = Arc::new(tokenizer);

    // 2. 出力ファイルの準備
    std::fs::create_dir_all(&args.output_dir)?;
    let train_path = args.output_dir.join("train.u32");
    let val_path = args.output_dir.join("val.u32");

    let mut train_writer = BufWriter::new(File::create(&train_path)?);
    let mut val_writer = BufWriter::new(File::create(&val_path)?);

    // 3. 入力ファイルを開く
    let file = File::open(&args.input).context("Failed to open corpus file")?;
    let reader = BufReader::new(file);

    // 行数カウント
    println!("📊 Counting lines for progress bar...");
    let total_lines = BufReader::new(File::open(&args.input)?).lines().count() as u64;

    let pb = ProgressBar::new(total_lines);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} lines ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    let mut chunk = Vec::with_capacity(args.batch_size);
    let mut total_tokens_train = 0usize;
    let mut total_tokens_val = 0usize;

    // EOSトークンID取得
    let eos_token = "<|endoftext|>";
    let eos_id = tokenizer
        .token_to_id(eos_token)
        .or_else(|| tokenizer.token_to_id("</s>"))
        .expect("EOS token (<|endoftext|> or </s>) not found in tokenizer.");

    println!("ℹ️ EOS Token ID: {}", eos_id);

    // 4. バッチ処理ループ
    for line_result in reader.lines() {
        let line = line_result?;
        chunk.push(line);

        if chunk.len() >= args.batch_size {
            let (t_count, v_count) = process_chunk(
                &chunk,
                &tokenizer,
                &mut train_writer,
                &mut val_writer,
                args.val_ratio,
                eos_id,
            )?;

            total_tokens_train += t_count;
            total_tokens_val += v_count;

            pb.inc(chunk.len() as u64);
            chunk.clear();
        }
    }

    if !chunk.is_empty() {
        let (t_count, v_count) = process_chunk(
            &chunk,
            &tokenizer,
            &mut train_writer,
            &mut val_writer,
            args.val_ratio,
            eos_id,
        )?;
        total_tokens_train += t_count;
        total_tokens_val += v_count;
        pb.inc(chunk.len() as u64);
    }

    pb.finish_with_message("Done");

    train_writer.flush()?;
    val_writer.flush()?;

    println!("✅ Processing Complete!");
    println!("   Train Tokens: {}", total_tokens_train);
    println!("   Val Tokens:   {}", total_tokens_val);
    println!("   Saved to:     {:?}", args.output_dir);

    Ok(())
}

fn process_chunk(
    lines: &[String],
    tokenizer: &Tokenizer,
    train_writer: &mut BufWriter<File>,
    val_writer: &mut BufWriter<File>,
    val_ratio: f64,
    eos_id: u32,
) -> Result<(usize, usize)> {
    let results: Vec<(Vec<u32>, bool)> = lines
        .par_iter()
        .map(|text| {
            let mut rng = rand::thread_rng();
            let is_val = rng.gen_bool(val_ratio);

            if let Ok(encoding) = tokenizer.encode(text.as_str(), false) {
                (encoding.get_ids().to_vec(), is_val)
            } else {
                (vec![], is_val)
            }
        })
        .collect();

    let mut t_count = 0;
    let mut v_count = 0;

    for (tokens, is_val) in results {
        if tokens.is_empty() {
            continue;
        }

        let target_writer = if is_val {
            v_count += tokens.len() + 1;
            &mut *val_writer
        } else {
            t_count += tokens.len() + 1;
            &mut *train_writer
        };

        for token in tokens {
            target_writer.write_u32::<LittleEndian>(token)?;
        }
        target_writer.write_u32::<LittleEndian>(eos_id)?;
    }

    Ok((t_count, v_count))
}
