use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, Optimizer, VarBuilder, VarMap};
use tokenizers::Tokenizer;

// Import our custom Candle module
// Import our custom Candle module
// #[path = "../core_engine.rs"]
// mod core_engine;
use cortex_rust::CandleTTTLayer as TTTLayer;

fn main() -> Result<()> {
    println!("--- Bit-TTT: First Utterance (Alice) ---");
    // 1. Setup Device (Auto-detect CUDA)
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Using Device: {:?}", device);

    // 2. Data Prep
    println!("Loading Tokenizer...");
    let tokenizer = Tokenizer::from_pretrained("gpt2", None).map_err(|e| anyhow::anyhow!(e))?;

    let text = "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversation?'";
    println!(
        "Training Text ({} chars): \"{}...\"",
        text.len(),
        &text[..50]
    );

    let tokens = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!(e))?
        .get_ids()
        .to_vec();
    println!("Total Tokens: {}", tokens.len());

    // 3. Model Setup
    // Trainable: Emb, Head
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let vocab_size = tokenizer.get_vocab_size(true);
    let hidden_dim = 64;

    let embedding = candle_nn::embedding(vocab_size, hidden_dim, vb.pp("emb"))?;
    let lm_head = candle_nn::linear(hidden_dim, vocab_size, vb.pp("head"))?;

    // Unfreezed: Bit-TTT Core
    // Now part of `vb` which comes from `varmap`
    let ttt_layer = TTTLayer::load(hidden_dim, 0.1, vb.pp("ttt"))?;
    let d_small = hidden_dim / 4;

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
    let epochs = 5; // FAST SAVE

    for epoch in 0..epochs {
        // Reset Hidden State (Fast Weight) at start of sequence
        let mut w_state = Tensor::zeros((1, d_small, d_small), DType::F32, &device)?;
        let mut total_loss = 0.0;

        // Sequence Loop
        for i in 0..tokens.len() - 1 {
            let input_token = tokens[i];
            let target_token = tokens[i + 1];

            // A. Embed
            let input_t = Tensor::new(&[input_token], &device)?;
            let x = embedding.forward(&input_t)?; // (1, D)

            // B. TTT Forward + Update
            let (out_feat, w_new) = ttt_layer.forward_update(&w_state, &x)?;
            w_state = w_new;

            // C. Residual
            let hidden = (out_feat + x)?;

            // D. Logits & Loss
            let logits = lm_head.forward(&hidden)?;
            let target_t = Tensor::new(&[target_token as i64], &device)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &target_t)?;

            total_loss += loss.to_scalar::<f32>()?;

            // E. Backward
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

    // 6. Generation (Multi-Prompt)
    println!("\n--- Generation Phase ---");
    let prompts = vec!["Alice"];

    for prompt in prompts {
        println!("\nPrompt: \"{}\"", prompt);
        let encoded = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!(e))?
            .get_ids()
            .to_vec();

        // Reset State for each prompt
        let mut w_state = Tensor::zeros((1, d_small, d_small), DType::F32, &device)?;
        let mut current_token = encoded[0];

        // Warmup: Feed prompt history
        for t in encoded.iter() {
            let input_t = Tensor::new(&[*t], &device)?;
            let x = embedding.forward(&input_t)?; // (1, D)
                                                  // Fix: w_new is the second return value
            let (_out_feat, w_new) = ttt_layer.forward_update(&w_state, &x)?;
            w_state = w_new;
            current_token = *t;
        }

        // Generate 20 tokens
        print!("{} -> ", prompt);
        for _ in 0..20 {
            let input_t = Tensor::new(&[current_token], &device)?;
            let x = embedding.forward(&input_t)?; // (1, D)

            // Forward (Update + Predict)
            // Fix: w_new is the second return value
            let (out_feat, w_new) = ttt_layer.forward_update(&w_state, &x)?;

            // Residual Connection
            let hidden = (out_feat + x)?;

            let logits = lm_head.forward(&hidden)?;
            let probs = candle_nn::ops::softmax(&logits, 1)?;
            let next_token = probs.argmax(1)?.squeeze(0)?.to_scalar::<u32>()?;

            let token_str = tokenizer.decode(&[next_token], true).unwrap_or("?".into());
            print!("{}", token_str);

            w_state = w_new;
            current_token = next_token;
        }
        println!("...");
    }
    println!("\n--- End Generation ---");

    Ok(())
}
