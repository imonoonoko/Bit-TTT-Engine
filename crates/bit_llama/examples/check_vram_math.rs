//! Example script to verify VRAM estimation math without GUI
//! Run with: cargo run --example check_vram_math

use bit_llama::config::ProjectConfig;

fn main() {
    println!("=== Bit-TTT VRAM Estimation Verification ===\n");

    // Case 1: Consumer Default (Small)
    let mut config = ProjectConfig::default();
    config.model_dim = 512;
    config.layers = 8;
    config.context_len = 512;
    print_metrics("Consumer Default (Dim=512, L=8)", &config);

    // Case 2: Llama-7B Equivalence (Approx)
    // Llama-2-7B: dim=4096, layers=32, vocab=32000
    config.model_dim = 4096;
    config.layers = 32;
    config.vocab_size = 32000;
    config.context_len = 2048;
    print_metrics("Llama-7B Class (Dim=4096, L=32)", &config);

    config.model_dim = 8192;
    config.layers = 80;
    config.n_heads = 64;
    config.n_kv_heads = Some(8); // GQA (Grouped Query Attention)
    config.context_len = 4096;
    print_metrics("Llama-70B Class (Dim=8192, L=80, GQA=8/64)", &config);
}

fn print_metrics(label: &str, config: &ProjectConfig) {
    let eff = config.estimate_efficiency();

    println!("Scenario: [{}]", label);
    println!(
        "  - Bit-TTT Usage : {:.2} MB ({:.2} GB)",
        eff.bit_ttt_mb,
        eff.bit_ttt_mb / 1024.0
    );
    println!(
        "  - Standard FP16 : {:.2} MB ({:.2} GB)",
        eff.fp16_mb,
        eff.fp16_mb / 1024.0
    );
    println!(
        "  - SAVED VRAM    : {:.2} MB ({:.2} GB)",
        eff.saved_mb,
        eff.saved_mb / 1024.0
    );
    println!("  - Efficiency    : {:.2}x", eff.saved_ratio);
    println!("  - Status        : {}", eff.status);
    println!("--------------------------------------------------");
}
