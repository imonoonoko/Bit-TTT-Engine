use anyhow::Result;
use byteorder::{LittleEndian, WriteBytesExt};
use clap::Args;
use flate2::read::GzDecoder;
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use minijinja::Environment;
use rand::Rng;
use rayon::prelude::*;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;
use zstd::stream::read::Decoder as ZstdDecoder;

#[derive(Args, Debug, Clone)]
pub struct PreprocessArgs {
    /// 入力コーパス (Glob pattern e.g. "data/*.jsonl")
    #[arg(long)]
    pub input: String,

    /// トークナイザー設定 (tokenizer.json)
    #[arg(long)]
    pub tokenizer: PathBuf,

    /// 出力ディレクトリ
    #[arg(long)]
    pub output_dir: PathBuf,

    /// Jinja2 Template (e.g. "User: {{instruction}}\nAI: {{output}}")
    #[arg(long)]
    pub template: Option<String>,

    /// JSON List Key (for huge JSON arrays)
    #[arg(long)]
    pub list_key: Option<String>,

    /// 検証データの割合 (0.01 = 1%)
    #[arg(long, default_value_t = 0.01)]
    pub val_ratio: f64,

    /// バッチサイズ (行数)。メモリに応じて調整。
    #[arg(long, default_value_t = 10_000)]
    pub batch_size: usize,
}

pub fn run(args: PreprocessArgs) -> Result<()> {
    println!("🚀 Starting Universal Preprocessing...");
    println!("   Input Pattern: {}", args.input);

    // 1. Setup Tokenizer
    let tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    let tokenizer = Arc::new(tokenizer);

    let eos_token = "<|endoftext|>";
    let eos_id = tokenizer
        .token_to_id(eos_token)
        .or_else(|| tokenizer.token_to_id("</s>"))
        .expect("EOS token (<|endoftext|> or </s>) not found.");
    println!("ℹ️ EOS Token ID: {}", eos_id);

    // Setup Jinja Env
    let mut env = Environment::new();
    let has_template = if let Some(tmpl) = &args.template {
        env.add_template("main", tmpl)?;
        println!("   Template: {}", tmpl);
        true
    } else {
        println!("   Mode: Raw Text (No template)");
        false
    };
    let env = Arc::new(env); // For Rayon sharing if needed (though we stream in main thread mostly)

    // 2. Output Setup
    std::fs::create_dir_all(&args.output_dir)?;
    let train_path = args.output_dir.join("train.u32");
    let val_path = args.output_dir.join("val.u32");
    let mut train_writer = BufWriter::new(File::create(&train_path)?);
    let mut val_writer = BufWriter::new(File::create(&val_path)?);

    // 3. Glob Expansion
    let paths_all: Vec<PathBuf> = glob(&args.input)?.filter_map(Result::ok).collect();
    // Filter only supported extensions, because glob crate doesn't support {ext,ext}
    let valid_exts = ["json", "jsonl", "txt", "md"];
    let paths: Vec<PathBuf> = paths_all
        .into_iter()
        .filter(|p| {
            if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
                valid_exts.contains(&ext)
            } else {
                false
            }
        })
        .collect();
    if paths.is_empty() {
        anyhow::bail!("No files found matching input pattern: {}", args.input);
    }
    println!("   Found {} files", paths.len());

    // 4. Processing Loop
    let mut chunk = Vec::with_capacity(args.batch_size);
    let mut total_tokens_train = 0usize;
    let mut total_tokens_val = 0usize;

    // Progress Bar (Total files)
    let pb = ProgressBar::new(paths.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} files ({msg})",
            )
            .unwrap(),
    );

    for path in paths {
        pb.set_message(
            path.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
        );

        let reader = open_compressed_file(&path)?;
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");

        if ext == "json" {
            // Full JSON Mode (Array or Object)
            // Note: This reads entire file into memory. For huge JSON files, use `serde_json::Deserializer::from_reader(reader).into_iter::<Value>()` stream.
            let stream = serde_json::Deserializer::from_reader(reader).into_iter::<Value>();

            for val_res in stream {
                match val_res {
                    Ok(val) => {
                        match val {
                            Value::Array(arr) => {
                                // Flatten array
                                for item in arr {
                                    chunk.push(item.to_string());
                                    if chunk.len() >= args.batch_size {
                                        let env_ref = if has_template { Some(&*env) } else { None };
                                        let (t, v) = process_chunk(
                                            &chunk,
                                            &tokenizer,
                                            &mut train_writer,
                                            &mut val_writer,
                                            args.val_ratio,
                                            eos_id,
                                            env_ref,
                                        )?;
                                        total_tokens_train += t;
                                        total_tokens_val += v;
                                        chunk.clear();
                                    }
                                }
                            }
                            _ => {
                                // Single Object
                                chunk.push(val.to_string());
                                if chunk.len() >= args.batch_size {
                                    let env_ref = if has_template { Some(&*env) } else { None };
                                    let (t, v) = process_chunk(
                                        &chunk,
                                        &tokenizer,
                                        &mut train_writer,
                                        &mut val_writer,
                                        args.val_ratio,
                                        eos_id,
                                        env_ref,
                                    )?;
                                    total_tokens_train += t;
                                    total_tokens_val += v;
                                    chunk.clear();
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("JSON Error in {:?}: {}", path, e);
                    }
                }
            }
        } else {
            // Line-based Mode (JSONL / TXT)
            let buffered = BufReader::new(reader);
            for line_res in buffered.lines() {
                let line = line_res?;
                if line.trim().is_empty() {
                    continue;
                }

                let text_to_process = if has_template {
                    if let Ok(_) = serde_json::from_str::<Value>(&line) {
                        // Render Template (Early render check not needed if we re-parse in process_chunk,
                        // but currently process_chunk re-parses.
                        // To avoid double parsing inefficiency we *could* pass Value,
                        // but to keep signature simple we pass String.
                        // Optimization: For JSONL we pass line as is? Yes.
                        line
                    } else {
                        // Not JSON? Skip?
                        continue;
                    }
                } else {
                    line
                };

                chunk.push(text_to_process);

                if chunk.len() >= args.batch_size {
                    let env_ref = if has_template { Some(&*env) } else { None };
                    let (t, v) = process_chunk(
                        &chunk,
                        &tokenizer,
                        &mut train_writer,
                        &mut val_writer,
                        args.val_ratio,
                        eos_id,
                        env_ref,
                    )?;
                    total_tokens_train += t;
                    total_tokens_val += v;
                    chunk.clear();
                }
            }
        }
        pb.inc(1);
    }

    // Flush remaining
    if !chunk.is_empty() {
        let env_ref = if has_template { Some(&*env) } else { None };
        let (t, v) = process_chunk(
            &chunk,
            &tokenizer,
            &mut train_writer,
            &mut val_writer,
            args.val_ratio,
            eos_id,
            env_ref,
        )?;
        total_tokens_train += t;
        total_tokens_val += v;
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

fn open_compressed_file(path: &Path) -> Result<Box<dyn Read + Send>> {
    let file = File::open(path)?;
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");

    match ext {
        "gz" => Ok(Box::new(GzDecoder::new(file))),
        "zst" => Ok(Box::new(ZstdDecoder::new(file)?)),
        _ => Ok(Box::new(file)),
    }
}

fn process_chunk(
    lines: &[String],
    tokenizer: &Tokenizer,
    train_writer: &mut BufWriter<File>,
    val_writer: &mut BufWriter<File>,
    val_ratio: f64,
    eos_id: u32,
    env: Option<&Environment>,
) -> Result<(usize, usize)> {
    let results: Vec<(Vec<u32>, bool)> = lines
        .par_iter()
        .map(|text| {
            let mut rng = rand::thread_rng();
            let is_val = rng.gen_bool(val_ratio);

            let text_to_tokenize = if let Some(env) = env {
                // Try Parse as JSON
                if let Ok(json_ctx) = serde_json::from_str::<Value>(text) {
                    // Render Template
                    match env
                        .get_template("main")
                        .and_then(|t| t.render(&json_ctx).map_err(|e| minijinja::Error::from(e)))
                    {
                        Ok(rendered) => rendered,
                        Err(_) => {
                            // Template or Template-lookup error: skip (empty) or raw?
                            // User wants fault tolerance: skipping is safer than garbage.
                            String::new()
                        }
                    }
                } else {
                    // Not valid JSON: skip or raw?
                    // If template is forced, and it's not JSON, it's noise.
                    String::new()
                }
            } else {
                // Raw Text Mode
                text.clone()
            };

            if text_to_tokenize.is_empty() {
                return (vec![], is_val);
            }

            if let Ok(encoding) = tokenizer.encode(text_to_tokenize.as_str(), false) {
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
