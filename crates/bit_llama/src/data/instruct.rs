use anyhow::Result;
use clap::{Args, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

#[derive(Debug, Deserialize, Serialize)]
pub struct InstructionEntry {
    pub instruction: String,
    pub input: String, // Can be empty
    pub output: String,
}

#[derive(Debug, Clone, ValueEnum, PartialEq)]
pub enum TemplateType {
    Alpaca,
    ChatML,
    Llama2,
    Raw,
}

#[derive(Debug, Clone)]
pub struct ChatTemplate {
    system_prompt: String,
    user_start: String,
    user_end: String,
    assistant_start: String,
    assistant_end: String,
}

impl ChatTemplate {
    pub fn from_type(t: TemplateType) -> Self {
        match t {
            TemplateType::Alpaca => Self {
                system_prompt: "".to_string(),
                user_start: "### Instruction:\n".to_string(),
                user_end: "\n".to_string(),
                assistant_start: "### Response:\n".to_string(),
                assistant_end: "".to_string(),
            },
            TemplateType::ChatML => Self {
                system_prompt: "".to_string(),
                user_start: "<|im_start|>user\n".to_string(),
                user_end: "<|im_end|>\n".to_string(),
                assistant_start: "<|im_start|>assistant\n".to_string(),
                assistant_end: "<|im_end|>\n".to_string(),
            },
            TemplateType::Llama2 => Self {
                system_prompt: "<<SYS>>\n".to_string(), // Simplified
                user_start: "[INST] ".to_string(),
                user_end: " [/INST] ".to_string(),
                assistant_start: "".to_string(),
                assistant_end: " </s>".to_string(),
            },
            TemplateType::Raw => Self {
                system_prompt: "".to_string(),
                user_start: "".to_string(),
                user_end: "".to_string(),
                assistant_start: "".to_string(),
                assistant_end: "".to_string(),
            },
        }
    }

    pub fn format(&self, entry: &InstructionEntry) -> (String, usize) {
        let mut full_text = String::new();

        // System Prompt (Optional handling, usually prepended if exists)
        if !self.system_prompt.is_empty() {
            full_text.push_str(&self.system_prompt);
        }

        // User Part (Instruction + Input)
        full_text.push_str(&self.user_start);
        full_text.push_str(&entry.instruction);
        if !entry.input.is_empty() {
            full_text.push('\n');
            full_text.push_str(&entry.input);
        }
        full_text.push_str(&self.user_end);

        // Assistant Part Start
        full_text.push_str(&self.assistant_start);

        // Boundary: This is where we start learning
        let response_start_idx = full_text.len();

        // Assistant Content
        full_text.push_str(&entry.output);
        full_text.push_str(&self.assistant_end);

        (full_text, response_start_idx)
    }
}

#[derive(Args, Debug, Clone)]
pub struct PrepareInstructArgs {
    /// Input JSON file (Alpaca format)
    #[arg(long)]
    pub input: String,

    /// Output directory for .u32 and .mask files
    #[arg(long, default_value = "workspace/data/instruct")]
    pub output: String,

    /// Path to tokenizer.json
    #[arg(long, default_value = "workspace/data/TinyStories/tokenizer.json")]
    pub tokenizer: String,

    /// Chat Template Type
    #[arg(long, value_enum, default_value_t = TemplateType::Alpaca)]
    pub template: TemplateType,
}

pub fn run(args: PrepareInstructArgs) -> Result<()> {
    println!("DEBUG: Starting prepare_instruct");
    println!("DEBUG: Input: {}", args.input);
    println!("DEBUG: Tokenizer: {}", args.tokenizer);
    let template = ChatTemplate::from_type(args.template);
    process_instruction_dataset(&args.input, &args.output, &args.tokenizer, template)
}

pub fn process_instruction_dataset(
    input_path: &str,
    output_dir: &str,
    tokenizer_path: &str,
    template: ChatTemplate,
) -> Result<()> {
    println!("Loading Tokenizer from: {}", tokenizer_path);
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    println!("Loading Dataset: {}", input_path);
    let file = File::open(input_path)?;
    let reader = BufReader::new(file);
    let entries: Vec<InstructionEntry> = serde_json::from_reader(reader)?;

    println!(
        "Found {} entries. Processing with template...",
        entries.len()
    );

    let mut all_tokens: Vec<u32> = Vec::new();
    let mut all_masks: Vec<u8> = Vec::new(); // 1 = Learn, 0 = Ignore

    let pb = ProgressBar::new(entries.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    for entry in entries {
        let (text, response_start_byte) = template.format(&entry);

        // Encode full text
        // We add special tokens (BOS) using the tokenizer's default behavior if configured
        let encoding = tokenizer
            .encode(text.clone(), true)
            .map_err(|e| anyhow::anyhow!(e))?;
        let ids = encoding.get_ids();
        let offsets = encoding.get_offsets();

        if ids.len() != offsets.len() {
            continue;
        }

        for (i, &token_id) in ids.iter().enumerate() {
            let (start_offset, _end_offset) = offsets[i];

            // Mask logic:
            // If the token starts BEFORE the response_start_byte, it's part of the prompt -> Mask = 0 (Ignore)
            let mask: u8 = if start_offset < response_start_byte {
                0
            } else {
                1
            };

            all_tokens.push(token_id);
            all_masks.push(mask);
        }

        pb.inc(1);
    }
    pb.finish_with_message("Tokenization complete");

    // Save .u32
    let output_path = Path::new(output_dir);
    std::fs::create_dir_all(output_path)?;

    let tokens_path = output_path.join("train.u32");
    let masks_path = output_path.join("train.mask");

    println!("Saving {} tokens to {:?}", all_tokens.len(), tokens_path);
    {
        let mut writer = BufWriter::new(File::create(&tokens_path)?);
        for &token in &all_tokens {
            writer.write_all(&token.to_le_bytes())?;
        }
    }

    println!("Saving masks to {:?}", masks_path);
    {
        let mut writer = BufWriter::new(File::create(&masks_path)?);
        writer.write_all(&all_masks)?; // u8, no endianness needed
    }

    println!("Only mask visual check (first 50 tokens):");
    for i in 0..50.min(all_tokens.len()) {
        let mask = all_masks[i];
        let status = if mask == 1 { "LEARN" } else { "SKIP " };
        println!(
            "[{:03}] ID:{:6} Mask:{} ({})",
            i, all_tokens[i], mask, status
        );
    }

    Ok(())
}
