use cortex_rust::TTTLayer;
use ndarray::Array2;
use rand::Rng;

fn main() {
    println!("=== Testing Bit-TTT Memory Capability (Rust) ===");

    // 1. Setup
    let dim = 64;
    // let seq_len = 20; // Handled by loop count
    let model = TTTLayer::new(dim, 0.1); // inner_lr = 0.1

    // 2. Create Pattern: 3 Distinct Vectors
    let mut rng = rand::thread_rng();
    let mut vocab = Array2::<f32>::zeros((3, dim));

    for i in 0..3 {
        for j in 0..dim {
            vocab[[i, j]] = rng.gen_range(-1.0..1.0);
        }
        // Normalize
        let row = vocab.row(i);
        let norm = row.dot(&row).sqrt();
        let mut row_mut = vocab.row_mut(i);
        row_mut.mapv_inplace(|v| v / norm);
    }

    // Sequence: 0, 1, 2, 0, 1, 2...
    let indices = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];

    println!("\nStep | Input | Reconstruction Loss (Before Update)");
    println!("---------------------------------------------");

    let mut w_state = Array2::<f32>::zeros((dim / 4, dim / 4));

    let mut losses_0 = Vec::new(); // Store losses for token '0'

    for (t, &idx) in indices.iter().enumerate() {
        let x_t = vocab.row(idx).to_owned();

        // Manual loop to peek inside, similar to Python script

        // 1. Projection
        let feat = model.proj_down.forward(&x_t);

        // 2. Predict (Before Update)
        let pred_feat = w_state.dot(&feat);

        // 3. Loss Calculation
        let error = &pred_feat - &feat;
        let loss = error.dot(&error) / (feat.len() as f32); // MSE

        // Store if it's token 0
        if idx == 0 {
            losses_0.push(loss);
        }

        println!("{:4} |   {}   | {:.6}", t, idx, loss);

        // 4. Update State (Re-using logic, slightly duplicative of Layer method but safer for visibility)
        let (w_new, _) = model.forward_state_update(&w_state, &x_t);
        w_state = w_new;
    }

    println!("---------------------------------------------");
    if losses_0.len() >= 2 {
        let first = losses_0[0];
        let second = losses_0[1];
        println!("Loss for '0' (1st time): {:.6}", first);
        println!("Loss for '0' (2nd time): {:.6}", second);

        if second < first {
            println!("\n[SUCCESS] Memory Effect Confirmed! Loss decreased.");
        } else {
            println!("\n[FAIL] No Memory Effect observed.");
        }
    }
}
