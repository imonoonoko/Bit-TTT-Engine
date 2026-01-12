//! Benchmark for BitLinearCpu
//! Measures throughput in GB/s and GOps/s

use candle_core::{Device, Tensor};
use cortex_rust::kernels::cpu::BitLinearCpu;
use cortex_rust::kernels::packing::PackedTensor;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("=== BitLinearCpu Kernel Benchmark ===");

    // Config: Simulate a typical layer (e.g. Llama-70B dimension)
    // Hidden Dim = 8192 (for 70B)
    // Batch Size = 1 (Inference)
    let m = 1;
    let k = 8192;
    let n = 8192;

    // Warmup & Stability
    let iterations = 100;

    // Device: CPU
    let device = Device::Cpu;

    println!("Configuration:");
    println!("  M (Batch): {}", m);
    println!("  K (Hidden): {}", k);
    println!("  N (Output): {}", n);
    println!("  Iterations: {}", iterations);

    // 1. Prepare Data
    println!("Preparing data (random float generation)...");
    let x_data = vec![0.5f32; m * k];
    let x = Tensor::from_vec(x_data, (m, k), &device)?;

    // Weights: Random {-1, 0, 1} pattern
    // We create a dummy PackedTensor directly to save setup time
    // Data size = N * K / 4
    let packed_len = n * k / 4;
    println!(
        "Packed Weight Size: {:.2} MB",
        packed_len as f64 / 1024.0 / 1024.0
    );

    // Fill with random bytes (simulating packed weights)
    // rand::random is slow, just fill cyclic pattern
    let w_data: Vec<u8> = (0..packed_len).map(|i| (i % 255) as u8).collect();

    let w_shape = candle_core::Shape::from((n, k));
    let packed_weights = PackedTensor::new(w_data, w_shape, 1.0, &device)?;

    // 2. Warmup
    println!("Warming up...");
    for _ in 0..10 {
        let _ = BitLinearCpu::forward(&x, &packed_weights)?;
    }

    // 3. Benchmark
    println!("Running benchmark...");
    let start = Instant::now();
    for _ in 0..iterations {
        // Use black_box to prevent compiler optimization (not strictly needed since result is used, but good practice if available)
        // Here we just accumulate result? No, just run.
        let out = BitLinearCpu::forward(&x, &packed_weights)?;
        // Force evaluation? Candle is lazy? Tensor ops are eager usually, except if graph.
        // BitLinearCpu::forward returns a Tensor. Data is computed eagerly in our CPU implementation.
        // But to be sure we touch memory?
        let _vec = out.flatten_all()?.to_vec1::<f32>()?;
    }
    let duration = start.elapsed();

    // 4. Report
    let total_secs = duration.as_secs_f64();
    let avg_sec = total_secs / iterations as f64;
    let avg_ms = avg_sec * 1000.0;

    // Ops: M * N * K * 2 (Add/Sub?)
    // Technically BitNet is Add/Sub, so 1 Op per weight access? Or 2 (Unpack + Add)?
    // Let's count "Effective MACs" = 2 * M * N * K
    let macs = (m as f64) * (n as f64) * (k as f64);
    let flops = macs / avg_sec; // "FLOPS" equivalent (actually IOPS)
    let gflops = flops / 1e9;

    // Memory Bandwidth:
    // Reads: X (M*K*4 bytes) + W (N*K/4 bytes)
    // Writes: Y (M*N*4 bytes)
    let bytes_read_x = (m * k * 4) as f64;
    let bytes_read_w = (packed_len) as f64; // 1 byte per 4 weights
    let bytes_write_y = (m * n * 4) as f64;
    let total_bytes = bytes_read_x + bytes_read_w + bytes_write_y;
    let gb_per_sec = (total_bytes / avg_sec) / 1e9;

    println!("\n=== Results ===");
    println!("Total Time: {:.4} s", total_secs);
    println!("Avg Latency: {:.4} ms / kernel", avg_ms);
    println!("Throughput: {:.2} GOps/s (Effective)", gflops);
    println!("Bandwidth:  {:.2} GB/s", gb_per_sec);

    Ok(())
}
