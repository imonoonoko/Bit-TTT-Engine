use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() -> anyhow::Result<()> {
    // Detect if we are building with CUDA feature or environment
    println!("cargo:rerun-if-changed=src/kernels/bit_op.cu");
    println!("cargo:rerun-if-changed=src/kernels/adaptive_bit_op.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/kernels/bit_op.ptx");
    println!("cargo:rerun-if-changed=src/kernels/adaptive_bit_op.ptx");

    let kernels = vec![
        ("src/kernels/bit_op.cu", "bit_op.ptx"),
        ("src/kernels/adaptive_bit_op.cu", "adaptive_bit_op.ptx"),
    ];

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    // Attempt to find NVCC
    let nvcc = match env::var("CUDA_HOME") {
        Ok(home) => PathBuf::from(home).join("bin/nvcc"),
        Err(_) => PathBuf::from("nvcc"),
    };

    // Check if nvcc works
    let is_cuda_available = Command::new(&nvcc).arg("--version").output().is_ok();

    for (cuda_file, ptx_filename) in kernels {
        let output_ptx_path = out_dir.join(ptx_filename);
        let saved_ptx_path = PathBuf::from("src/kernels").join(ptx_filename);

        if is_cuda_available {
            let output = Command::new(&nvcc)
                .arg("-ptx")
                // Use compute_75 as baseline per recommendation (Turing)
                // But keep compute_61 for older pascal support if needed?
                // Providing multiple architectures in PTX is tricky via simple CLI.
                // We'll stick to a reasonable robust default.
                .arg("-arch=compute_75")
                .arg("-code=sm_75")
                .arg(cuda_file)
                .arg("-o")
                .arg(&output_ptx_path)
                .output();

            match output {
                Ok(out) if out.status.success() => {
                    let _ = fs::copy(&output_ptx_path, &saved_ptx_path);
                    println!("cargo:warning=Updated bundled PTX at {:?}", saved_ptx_path);
                }
                Ok(out) => {
                    let err = String::from_utf8_lossy(&out.stderr);
                    println!(
                        "cargo:warning=CUDA compilation failed for {} (using fallback): {}",
                        cuda_file, err
                    );
                }
                Err(e) => {
                    println!("cargo:warning=Failed to execute NVCC: {}", e);
                }
            }
        }

        // Fallback checks
        if !output_ptx_path.exists()
            || output_ptx_path
                .metadata()
                .map(|m| m.len() == 0)
                .unwrap_or(true)
        {
            if saved_ptx_path.exists() {
                println!(
                    "cargo:warning=Using bundled PTX from {:?} for {}",
                    saved_ptx_path, ptx_filename
                );
                fs::copy(&saved_ptx_path, &output_ptx_path)?;
            } else {
                println!(
                    "cargo:warning=CRITICAL: No PTX found for {}. Feature will fail.",
                    ptx_filename
                );
                fs::write(&output_ptx_path, "")?;
            }
        }
    }

    Ok(())
}
