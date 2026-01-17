use anyhow::Result;
use candle_core::Tensor;
use clap::Args;
use cortex_rust::Llama;
// use memmap2::MmapOptions; // Removed
// use std::fs::File; // Removed
use crate::loader::BitLoader;
use std::io::{self, Write};
use tracing::{info, warn};

#[derive(Args, Debug, Clone)]
pub struct EvaluateArgs {
    #[arg(short, long, default_value = ".")]
    pub model: String,

    #[arg(short, long, required = true)]
    pub data: String,

    #[arg(long, default_value_t = 128)]
    pub context_len: usize,

    #[arg(long, default_value_t = 8)]
    pub batch_size: usize,

    #[arg(long)]
    pub limit: Option<usize>,
}

pub fn run(args: EvaluateArgs) -> Result<()> {
    info!("--- Bit-Llama Evaluation (Perplexity) ---");
    info!("Model: {}", args.model);
    info!("Data:  {}", args.data);

    let mut llama = Llama::load_auto(&args.model)?;
    llama.model.precompute_packed()?;
    info!("Model loaded successfully on {:?}", llama.device);

    let mut loader = BitLoader::new(&args.data)?;
    // Disable looping for evaluation
    loader = loader.with_loop(false);

    info!("Data loaded. Total tokens: {}", loader.data_len);

    let mut total_nll = 0.0;
    let mut total_tokens = 0;
    let mut batch_count = 0;

    let _d_small = llama.model.config.hidden_dim / 4;
    let _num_layers = llama.model.config.num_layers;

    info!("Starting Evaluation...");

    loop {
        match loader.next_batch(args.batch_size, args.context_len, &llama.device) {
            Ok((input, target)) => {
                let (b_sz, seq_len) = input.dims2()?;
                let input_vec = input.to_vec2::<u32>()?;
                let target_vec = target.to_vec2::<u32>()?; // BitLoader returns Targets now too!

                // Note: BitLoader returns targets as well. The original EvalLoader also did basically same slice.
                // We can use the target tensor directly.
                // But wait, the original evaluate code iterates token by token inside batch for causality check?
                // No, it's just standard NLL calc.
                // "Current TTT impl in loop is explicit content from original."

                for b in 0..b_sz {
                    let mut w_states = llama.model.new_w_states();

                    // Reset KV Cache and Position for new sequence
                    llama.model.reset_kv_cache();

                    let mut batch_nll = 0.0;

                    for t in 0..seq_len {
                        let token_id = input_vec[b][t];
                        let target_id = target_vec[b][t];

                        let inp_t = Tensor::new(&[token_id], &llama.device)?;
                        let logits = llama.model.forward_one(&inp_t, &mut w_states)?;
                        if b == 0 && t == 0 {
                            eprintln!(
                                "🚀 [DEBUG] Starting loop. Logits shape: {:?}",
                                logits.shape()
                            );
                        }

                        if b == 0 && t < 3 {
                            // Safe print attempt
                            // let l_vec = logits.to_vec2::<f32>()?;
                            // eprintln!("🔍 [LOGITS] t={}: {:?}", t, &l_vec[0][..10]);
                        }

                        let tgt_t = Tensor::new(&[target_id as i64], &llama.device)?;

                        // Let's just reshape explicitly to be safe.
                        let (_b, _s, vocab) = logits.dims3()?;
                        let logits_2d = logits.reshape((1, vocab))?;

                        let loss = candle_nn::loss::cross_entropy(&logits_2d, &tgt_t)?;
                        batch_nll += loss.to_scalar::<f32>()?;
                    }

                    total_nll += batch_nll;
                    total_tokens += seq_len;
                }

                batch_count += 1;
                if batch_count % 10 == 0 {
                    print!(".");
                    io::stdout().flush()?;
                }

                if let Some(limit) = args.limit {
                    if total_tokens >= limit {
                        info!("\n[Limit Reached] Stopping.");
                        break;
                    }
                }
            }
            Err(e) => {
                // Assuming "End of Data" error stops loop
                if e.to_string().contains("End of Data") {
                    break;
                } else if e.to_string().contains("Shape mismatch") {
                    // Candle error on partial batch?
                    warn!("\nStopping due to partial batch or error: {}", e);
                    break;
                } else {
                    // Log other errors but maybe break too
                    warn!("\nError during evaluation: {}", e);
                    break;
                }
            }
        }
    }

    println!(); // Newline after dots
    if total_tokens > 0 {
        let avg_nll = total_nll / total_tokens as f32;
        let ppl = avg_nll.exp();
        info!("--------------------------------");
        info!("Total Tokens: {}", total_tokens);
        info!("Avg NLL:      {:.4}", avg_nll);
        info!("Perplexity:   {:.2}", ppl);
        info!("--------------------------------");
    } else {
        warn!("No data processed.");
    }

    Ok(())
}
