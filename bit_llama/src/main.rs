use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Module};
use tokenizers::Tokenizer;

mod core_engine;
use core_engine::TTTLayer;

fn main() -> Result<()> {
    println!("--- Bit-TTT Rust Native Experiment (Candle Core) ---");

    // 1. Setup Device (Auto-detect CUDA)
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Using Device: {:?}", device);

    // 2. Load Tokenizer (GPT-2)
    println!("Loading Tokenizer...");
    let tokenizer = Tokenizer::from_pretrained("gpt2", None).map_err(|e| anyhow::anyhow!(e))?;

    // 3. Define Input Text (Alice in Wonderland snippet for Scaling Test)
    let text = "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversation?'";
    println!("Input Text: \"{}...\"", &text[..50]);

    // 4. Tokenize
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let tokens = encoding.get_ids();
    println!("Tokens: {:?}", tokens);

    // 5. Create Dummy Embeddings
    let vocab_size = tokenizer.get_vocab_size(true);
    let hidden_dim = 64;
    let d_small = hidden_dim / 4;

    let embedding_weights = Tensor::randn(0f32, 1.0, (vocab_size, hidden_dim), &device)?;
    let embeddings = Embedding::new(embedding_weights, hidden_dim);

    // 6. Initialize Bit-TTT Layer
    // NEW: Use VarBuilder with Zeros for initialization
    let varchar = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varchar, DType::F32, &device);
    let ttt_layer = TTTLayer::load(hidden_dim, 0.1, vb)?;

    // Initialize Hidden State (W_state) - Tensor directly
    // NEW: Use Batch Dimension (1, d_small, d_small)
    let mut w_state = Tensor::zeros((1, d_small, d_small), DType::F32, &device)?;

    println!("\n--- Starting Online Learning ---");
    println!("Step | TokenID | Status");
    println!("--------------------------------------");

    // 7. Loop through tokens
    for (i, &token_id) in tokens.iter().enumerate() {
        // A. Get Embedding: (1, Hidden)
        let input_tensor = Tensor::new(&[token_id], &device)?;
        let embedding_vector = embeddings.forward(&input_tensor)?;
        // Shape is already (1, Hidden) from embedding.forward with (1) input

        // B. Update Step (Forward + Train)
        // Returns (out_feat, w_new)
        let (_out_feat, w_new) = ttt_layer.forward_update(&w_state, &embedding_vector)?;

        // Update state
        w_state = w_new;

        // D. Log
        let token_str = tokenizer
            .decode(&[token_id], true)
            .unwrap_or_else(|_| "???".to_string());
        println!("{:4} | {:<7} ({:<5}) | Updated", i, token_str, token_id);
    }

    println!("\n✅ Experiment Finished.");
    Ok(())
}
