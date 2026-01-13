use anyhow::Result;
use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};
use std::path::Path;

use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDec;
use tokenizers::decoders::metaspace::Metaspace as MetaspaceDec;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::models::unigram::{Unigram, UnigramTrainerBuilder};
use tokenizers::normalizers::unicode::NFKC;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::metaspace::Metaspace;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::byte_level::ByteLevel as ByteLevelPost;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::{AddedToken, TokenizerImpl};

use crate::data::sampler::ParallelSampler;

#[derive(Debug, Clone, ValueEnum, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ModelType {
    Bpe,
    #[default]
    Unigram,
}

#[derive(Args, Debug, Clone)]
pub struct VocabArgs {
    #[arg(long, default_value = "data/Wiki40b/corpus.txt")]
    pub input: String,

    #[arg(long, default_value = "data/Wiki40b/tokenizer.json")]
    pub output: String,

    #[arg(long, default_value_t = 48000)]
    pub vocab_size: usize,

    #[arg(long, default_value_t = 2)]
    pub min_frequency: u64,

    #[arg(long, value_enum, default_value_t = ModelType::Unigram)]
    pub model_type: ModelType,

    /// Limit input to first N Megabytes (Optimized for speed)
    #[arg(long)]
    pub limit_mb: Option<usize>,
}

pub fn run(args: VocabArgs) -> Result<()> {
    match args.model_type {
        ModelType::Bpe => train_bpe(args),
        ModelType::Unigram => train_unigram(args),
    }
}

fn prepare_files(args: &VocabArgs) -> Result<Vec<String>> {
    println!("--- Rust Tokenizer Trainer ---");
    println!("Input:      {}", args.input);
    println!("Output:     {}", args.output);
    println!("Vocab Size: {}", args.vocab_size);
    println!("Type:       {:?}", args.model_type);

    if let Some(parent) = Path::new(&args.output).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let files: Vec<String> = glob::glob(&args.input)?
        .filter_map(|x| x.ok())
        .map(|p| p.to_string_lossy().to_string())
        .collect();

    let files_to_train = if files.is_empty() {
        if Path::new(&args.input).exists() {
            println!("Training on single file: {}", args.input);
            vec![args.input.clone()]
        } else {
            anyhow::bail!("No input files found at: {}", args.input);
        }
    } else {
        println!("Training on {} files...", files.len());
        files
    };
    if let Some(limit_mb) = args.limit_mb {
        let sample_path = Path::new(&args.output).parent().unwrap().join("corpus_sample.txt");
        return ParallelSampler::sample(files_to_train, sample_path, limit_mb);
    }

    Ok(files_to_train)
}

fn get_special_tokens() -> Vec<AddedToken> {
    vec![
        AddedToken::from(String::from("<|endoftext|>"), true),
        AddedToken::from(String::from("<|padding|>"), true),
        AddedToken::from(String::from("<|user|>"), true),
        AddedToken::from(String::from("<|model|>"), true),
        AddedToken::from(String::from("<|system|>"), true),
        AddedToken::from(String::from("<UNK>"), true),
    ]
}

fn train_bpe(args: VocabArgs) -> Result<()> {
    let files = prepare_files(&args)?;
    println!("Configuration: BPE (GPT-2 Style)");

    // Explicitly typed TokenizerImpl for BPE
    let mut tokenizer: TokenizerImpl<
        BPE,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    > = TokenizerImpl::new(BPE::default());
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::ByteLevel(ByteLevel::default())));
    tokenizer.with_decoder(Some(DecoderWrapper::ByteLevel(ByteLevelDec::default())));
    tokenizer.with_post_processor(Some(PostProcessorWrapper::ByteLevel(ByteLevelPost::default())));
    tokenizer.with_normalizer(Option::<NormalizerWrapper>::None);

    let special_tokens = get_special_tokens();

    let mut trainer = BpeTrainerBuilder::new()
        .vocab_size(args.vocab_size)
        .min_frequency(args.min_frequency)
        .special_tokens(special_tokens)
        .build();

    println!("Starting training (BPE)...");
    tokenizer.train_from_files(&mut trainer, files).map_err(|e| anyhow::anyhow!("{}", e))?;

    tokenizer.save(&args.output, true).map_err(|e| anyhow::anyhow!("{}", e))?;
    println!("✅ Tokenizer saved to {}", args.output);
    println!("📊 Final vocab size: {}", tokenizer.get_vocab_size(true));
    Ok(())
}

fn train_unigram(args: VocabArgs) -> Result<()> {
    let files = prepare_files(&args)?;
    println!("Configuration: Unigram (SentencePiece Style) + NFKC");

    // Explicitly typed TokenizerImpl for Unigram
    let mut tokenizer: TokenizerImpl<
        Unigram,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    > = TokenizerImpl::new(Unigram::default());

    // Unigram Setup: NFKC -> Metaspace
    tokenizer.with_normalizer(Some(NormalizerWrapper::NFKC(NFKC)));
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Metaspace(Metaspace::default())));
    tokenizer.with_decoder(Some(DecoderWrapper::Metaspace(MetaspaceDec::default())));

    let special_tokens = get_special_tokens();

    let mut trainer = UnigramTrainerBuilder::default()
        .vocab_size(args.vocab_size as u32)
        .special_tokens(special_tokens)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build Unigram trainer: {}", e))?;

    println!("Starting training (Unigram)...");
    tokenizer.train_from_files(&mut trainer, files).map_err(|e| anyhow::anyhow!("{}", e))?;

    tokenizer.save(&args.output, true).map_err(|e| anyhow::anyhow!("{}", e))?;
    println!("✅ Tokenizer saved to {}", args.output);
    println!("📊 Final vocab size: {}", tokenizer.get_vocab_size(true));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tokenizers::Tokenizer; // Use dynamic Tokenizer for loading/verification

    #[test]
    fn test_unigram_training_and_normalization() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let input_path = dir.path().join("corpus_ja.txt");
        let output_path = dir.path().join("tokenizer.json");

        // Create dummy Japanese corpus with mixed characters
        let mut file = std::fs::File::create(&input_path)?;
        writeln!(file, "昔々あるところに、お爺さんとお婆さんが住んでいました。")?;
        writeln!(file, "ＡＢＣａｂｃ１２３")?; // Fullwidth to test NFKC
        writeln!(file, "🍎🍊🍇")?; // Emojis
        writeln!(file, "Kyoto is nice.")?;

        let args = VocabArgs {
            input: input_path.to_string_lossy().to_string(),
            output: output_path.to_string_lossy().to_string(),
            vocab_size: 100, // Small vocab for small data
            min_frequency: 1,
            model_type: ModelType::Unigram,
            limit_mb: None,
        };

        // Run training
        run(args)?;

        // Load and verify
        let tokenizer = Tokenizer::from_file(&output_path).map_err(|e| anyhow::anyhow!(e))?;

        // Test Normalization (NFKC: ＡＢＣ -> ABC)
        let encoding = tokenizer.encode("ＡＢＣ", false).map_err(|e| anyhow::anyhow!(e))?;
        let tokens = encoding.get_tokens();
        println!("Normalized Tokens: {:?}", tokens);

        // In Unigram with Metaspace, "ABC" typically becomes " ABC" (with underscore)
        // We just check that it's NOT fullwidth anymore
        assert!(tokens.iter().any(|t| t.contains("ABC") || t.contains("A")));
        assert!(!tokens.iter().any(|t| t.contains("Ａ")));

        // Test Roundtrip
        let text = "昔々あるところに";
        let encoded = tokenizer.encode(text, false).map_err(|e| anyhow::anyhow!(e))?;
        let decoded = tokenizer.decode(encoded.get_ids(), false).map_err(|e| anyhow::anyhow!(e))?;
        // Remove Metaspace underscores (U+2581) for comparison
        assert_eq!(decoded.replace("\u{2581}", ""), text);

        Ok(())
    }
}
