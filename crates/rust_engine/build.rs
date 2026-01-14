use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() -> anyhow::Result<()> {
    // Detect if we are building with CUDA feature or environment
    println!("cargo:rerun-if-changed=src/kernels/bit_op.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/kernels/bit_op.ptx");

    let cuda_file = "src/kernels/bit_op.cu";
    let ptx_filename = "bit_op.ptx";

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let output_ptx_path = out_dir.join(ptx_filename);

    // The source of truth for the bundled PTX (committed to git)
    let saved_ptx_path = PathBuf::from("src/kernels").join(ptx_filename);

    // Attempt to find NVCC
    let nvcc = match env::var("CUDA_HOME") {
        Ok(home) => PathBuf::from(home).join("bin/nvcc"),
        Err(_) => PathBuf::from("nvcc"),
    };

    // Check if nvcc works
    let is_cuda_available = Command::new(&nvcc).arg("--version").output().is_ok();

    if is_cuda_available {
        // Try to compile
        // Note: For complex environments (MSVC), nvcc might fail if cl.exe is not in PATH.
        // build.rs runs in the cargo environment, which might not have vcvars set up for shell usage.
        let output = Command::new(&nvcc)
            .arg("-ptx")
            .arg("-arch=compute_80") // Target Ampere (RTX 30 series+) or adjust
            .arg("-code=sm_80")
            .arg(cuda_file)
            .arg("-o")
            .arg(&output_ptx_path)
            .output();

        match output {
            Ok(out) if out.status.success() => {
                // Compilation Success!
                // Update the bundled PTX so we can commit it.
                // Note: Modifying src/ during build is generally discouraged, but necessary here
                // to act as a "cache" for non-CUDA users.
                // We only do this if we actually compiled successfully.
                let _ = fs::copy(&output_ptx_path, &saved_ptx_path);
                println!("cargo:warning=Updated bundled PTX at {:?}", saved_ptx_path);
            }
            Ok(out) => {
                let err = String::from_utf8_lossy(&out.stderr);
                println!(
                    "cargo:warning=CUDA compilation failed (using fallback): {}",
                    err
                );
            }
            Err(e) => {
                println!("cargo:warning=Failed to execute NVCC: {}", e);
            }
        }
    } else {
        println!("cargo:warning=NVCC not found. Skipping compilation.");
    }

    // Fallback: If output_ptx doesn't exist (or is empty), use the bundled one
    if !output_ptx_path.exists()
        || output_ptx_path
            .metadata()
            .map(|m| m.len() == 0)
            .unwrap_or(true)
    {
        if saved_ptx_path.exists() {
            println!(
                "cargo:warning=Using bundled PTX from {:?} (NVCC missing or failed).",
                saved_ptx_path
            );
            fs::copy(&saved_ptx_path, &output_ptx_path)?;
        } else {
            // No bundle, no compiler. Create meaningful dummy or fail.
            // Creating empty dummy allows build to pass, but runtime might panic.
            println!("cargo:warning=CRITICAL: No PTX found. Feature requiring CUDA will fail.");
            fs::write(&output_ptx_path, "")?;
        }
    }

    Ok(())
}
