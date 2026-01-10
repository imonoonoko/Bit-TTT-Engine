use anyhow::Result;
use clap::Args;
use std::path::Path;

use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDec;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::byte_level::ByteLevel as ByteLevelPost;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::{AddedToken, TokenizerImpl};

#[derive(Args, Debug, Clone)]
pub struct VocabArgs {
    #[arg(long, default_value = "data/Wiki40b/corpus.txt")]
    pub input: String,

    #[arg(long, default_value = "data/Wiki40b/tokenizer.json")]
    pub output: String,

    #[arg(long, default_value_t = 32000)]
    pub vocab_size: usize,

    #[arg(long, default_value_t = 2)]
    pub min_frequency: u64,
}

type MyTokenizer = TokenizerImpl<
    BPE,
    tokenizers::normalizers::NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

pub fn run(args: VocabArgs) -> Result<()> {
    println!("--- Rust Tokenizer Trainer ---");
    println!("Input:      {}", args.input);
    println!("Output:     {}", args.output);
    println!("Vocab Size: {}", args.vocab_size);
    println!("Min Freq:   {}", args.min_frequency);

    if let Some(parent) = Path::new(&args.output).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut tokenizer: MyTokenizer = TokenizerImpl::new(BPE::default());

    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::ByteLevel(ByteLevel::default())));
    tokenizer.with_decoder(Some(DecoderWrapper::ByteLevel(ByteLevelDec::default())));
    tokenizer.with_post_processor(Some(PostProcessorWrapper::ByteLevel(
        ByteLevelPost::default(),
    )));
    tokenizer.with_normalizer(Option::<tokenizers::normalizers::NormalizerWrapper>::None);

    let special_tokens = vec![
        AddedToken::from(String::from("<|endoftext|>"), true),
        AddedToken::from(String::from("<|padding|>"), true),
    ];

    let mut trainer = BpeTrainerBuilder::new()
        .vocab_size(args.vocab_size)
        .min_frequency(args.min_frequency)
        .special_tokens(special_tokens)
        .build();

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

    println!("Starting training...");

    tokenizer
        .train_from_files(&mut trainer, files_to_train)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    tokenizer
        .save(&args.output, true)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    println!("✅ Tokenizer saved to {}", args.output);
    println!("📊 Final vocab size: {}", tokenizer.get_vocab_size(true));

    Ok(())
}
