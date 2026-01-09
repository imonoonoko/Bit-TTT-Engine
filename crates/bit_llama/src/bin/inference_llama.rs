use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use std::io::{self, Write};
use std::path::Path;
use tokenizers::Tokenizer;

use cortex_rust::{BitLlama, BitLlamaConfig};

// ============================================================
// Model Configuration Constants
// ============================================================
/// Model hidden dimension (embedding size).
const DIM: usize = 256;

/// Number of transformer blocks (layers).
const LAYERS: usize = 4;

/// Vocabulary size. Must match the BPE tokenizer.
const VOCAB: usize = 16384;

fn main() -> Result<()> {
    println!("--- Bit-Llama Inference ---");
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);

    // 1. Load Tokenizer
    let tokenizer_path = "data/TinyStories/tokenizer.json";
    if !Path::new(tokenizer_path).exists() {
        anyhow::bail!("Tokenizer not found: {}", tokenizer_path);
    }
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;

    // 2. Load Model
    let config = BitLlamaConfig {
        vocab_size: VOCAB,
        hidden_dim: DIM,
        num_layers: LAYERS,
        inner_lr: 0.1,
    };

    let weight_path = "bit_llama_checkpoint.safetensors";
    // Check if checkpoint exists, otherwise look for final
    let weight_path = if Path::new(weight_path).exists() {
        weight_path
    } else if Path::new("bit_llama_v1.safetensors").exists() {
        "bit_llama_v1.safetensors"
    } else {
        anyhow::bail!("No model weights found! Run training first.");
    };

    println!("Loading weights from: {}", weight_path);
    println!("Loading weights from: {}", weight_path);
    // Safety: We assume the safetensors file is not modified while mapped.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weight_path], DType::F32, &device)? };
    let model = BitLlama::load(config, vb)?;

    println!("Model Loaded!");

    // 3. Interactive Loop
    let mut w_states = Vec::new(); // Persistent State (Optionally reset per prompt)

    loop {
        print!("\n> ");
        io::stdout().flush()?;
        let mut prompt = String::new();
        io::stdin().read_line(&mut prompt)?;
        let prompt = prompt.trim();
        if prompt.is_empty() {
            continue;
        }
        if prompt == "exit" || prompt == "quit" {
            break;
        }

        // Reset State for new story? For TTT, actually we can keep it (Long-term memory),
        // but for TinyStories usually we want fresh context.
        // Let's reset for now to be safe.
        // w_states: (1, d_small, d_small) for Batch=1
        w_states.clear();
        let d_small = DIM / 4;
        for _ in 0..LAYERS {
            w_states.push(Tensor::zeros((1, d_small, d_small), DType::F32, &device)?);
        }

        // Encode
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!(e))?;
        let input_ids = encoding.get_ids();
        let mut current_ids = input_ids.to_vec();

        print!("{}", prompt);
        io::stdout().flush()?;

        // Prefill (Process Prompt)
        for &id in input_ids {
            let input_tensor = Tensor::new(&[id], &device)?.unsqueeze(0)?; // (1, 1)

            // Manual Forward for Batch 1
            let mut h = model.embedding.forward(&input_tensor)?.squeeze(1)?; // (1, D)

            for (i, layer) in model.layers.iter().enumerate() {
                let (h_new, w_new) = layer.forward(&h, &w_states[i])?;
                h = h_new;
                w_states[i] = w_new;
            }
        }

        // Generate
        let max_gen_tokens = 100;
        let mut last_token_id = *input_ids.last().unwrap_or(&0);

        for _ in 0..max_gen_tokens {
            let input_tensor = Tensor::new(&[last_token_id], &device)?.unsqueeze(0)?; // (1, 1)

            // Manual Forward
            let mut h = model.embedding.forward(&input_tensor)?.squeeze(1)?; // (1, D)

            for (i, layer) in model.layers.iter().enumerate() {
                let (h_new, w_new) = layer.forward(&h, &w_states[i])?;
                h = h_new;
                w_states[i] = w_new;
            }

            let h_norm = model.norm.forward(&h)?;
            let logits = model.lm_head.forward(&h_norm)?; // (1, V)

            // Greedy Sampling
            let logits_v: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
            let next_token = logits_v
                .iter()
                .enumerate()
                .max_by(|(_, a): &(usize, &f32), (_, b): &(usize, &f32)| a.total_cmp(b))
                .map(|(index, _)| index as u32)
                .ok_or_else(|| anyhow::anyhow!("Logits empty"))?;

            last_token_id = next_token;
            current_ids.push(next_token);

            // Decode & Print
            let decoded = tokenizer
                .decode(&[next_token], false)
                .map_err(|e| anyhow::anyhow!(e))?;
            print!("{}", decoded);
            io::stdout().flush()?;

            if next_token == tokenizer.token_to_id("<|endoftext|>").unwrap_or(0) {
                break;
            }
        }
        println!("\n");
    }

    Ok(())
}
