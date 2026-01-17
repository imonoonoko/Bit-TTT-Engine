use candle_core::{DType, Device, Result};
use candle_nn::VarBuilder;

fn main() -> Result<()> {
    let path = "models/TinyLlama-Adaptive-Converted/model.safetensors";
    let device = Device::Cpu;

    // Load VB
    unsafe {
        let vb = VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device)?;
        println!("Loaded VB.");

        let tensor_name = "model.layers.0.mlp.gate_proj.weight_packed";

        // Try to get metadata/shape? VarBuilder doesn't expose easy metadata inspection directly without loading.
        // Try loading with expected shape
        let shape = (5632, 512, 3);
        match vb.get(shape, tensor_name) {
            Ok(t) => println!("✅ Loaded successfully: {:?}", t),
            Err(e) => println!("❌ Failed to load [5632, 512, 3]: {:?}", e),
        }

        // Try loading with 1 base shape
        let shape1 = (5632, 512, 1);
        match vb.get(shape1, tensor_name) {
            Ok(t) => println!("❓ Loaded [5632, 512, 1]: {:?}", t),
            Err(e) => println!("❌ Failed to load [5632, 512, 1]: {:?}", e),
        }

        // Try loading flat
        // 5632 * 512 * 3 = 8650752
        match vb.get((8650752,), tensor_name) {
            Ok(t) => println!("✅ Loaded flat 3-base: {:?}", t),
            Err(e) => println!("❌ Failed to load flat 3-base: {:?}", e),
        }

        // 5632 * 512 = 2883584
        match vb.get((2883584,), tensor_name) {
            Ok(t) => println!("❓ Loaded flat 1-base: {:?}", t),
            Err(e) => println!("❌ Failed to load flat 1-base: {:?}", e),
        }
    }

    Ok(())
}
