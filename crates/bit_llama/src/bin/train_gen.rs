use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use cortex_rust::{BitLlama, BitLlamaConfig};
use tokenizers::Tokenizer;

// Uses BitLlama structure for training (instead of manual layer implementation)

fn main() -> Result<()> {
    println!("--- Bit-TTT: Training (New Ecosystem) ---");

    // 1. Setup Device
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Using Device: {:?}", device);

    // 2. Data Prep
    // 2. Data Prep
    println!("Loading Tokenizer from local dummy...");
    let tokenizer_path = std::path::Path::new("../models/dummy/tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?
    } else {
        println!("Local tokenizer not found, trying download (gpt2)...");
        Tokenizer::from_pretrained("gpt2", None).map_err(|e| anyhow::anyhow!(e))?
    };

    let text = "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversation?'";
    println!("Training Text: \"{}...\"", &text[..50]);

    let tokens = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!(e))?
        .get_ids()
        .to_vec();
    println!("Total Tokens: {}", tokens.len());

    // 3. Model Setup (Trainable)
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let config = BitLlamaConfig {
        vocab_size: tokenizer.get_vocab_size(true),
        hidden_dim: 64,
        num_layers: 2,
        inner_lr: 0.001,
    };

    println!("Initializing BitLlama Model...");
    let model = BitLlama::load(config, vb)?;

    // 4. Optimizer
    let mut adam = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            lr: 0.005,
            ..Default::default()
        },
    )?;

    // 5. Training Loop
    println!("\n--- Training Start ---");
    let epochs = 5;
    let d_small = config.hidden_dim / 4;

    for epoch in 0..epochs {
        // Init Hidden States (Fast Weights)
        // Note: Llama::new does this internally for inference, but here we manage it manually for training loop
        let mut w_states = Vec::new();
        for _ in 0..config.num_layers {
            let w = Tensor::zeros((d_small, d_small), DType::F32, &device)?;
            w_states.push(w);
        }

        let mut total_loss = 0.0;

        // Sequence Loop
        for i in 0..tokens.len() - 1 {
            let input_token = tokens[i];
            let target_token = tokens[i + 1];

            // A. Input Tensor
            let input_t = Tensor::new(&[input_token], &device)?; // (1)

            // B. Forward (BitLlama)
            // forward_one returns logits: (V) or (1, V)
            // BitLlama::forward_one modifies w_states in-place (TTT update)
            let logits = model.forward_one(&input_t, &mut w_states)?;

            // C. Loss
            let target_t = Tensor::new(&[target_token as i64], &device)?;
            // forward_one returns 1D logits [Vocab], but cross_entropy needs [Batch, Vocab]
            let logits_batch = logits.unsqueeze(0)?;
            let loss = candle_nn::loss::cross_entropy(&logits_batch, &target_t)?;

            total_loss += loss.to_scalar::<f32>()?;

            // D. Backward
            adam.backward_step(&loss)?;
        }

        println!(
            "Epoch {:3} | Avg Loss: {:.5}",
            epoch,
            total_loss / (tokens.len() as f32)
        );
    }

    // SAVE BRAIN
    println!("\nSaving Brain to alice_brain.safetensors...");
    varmap.save("alice_brain.safetensors")?;

    println!("\n--- End Training ---");
    Ok(())
}
