use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() -> anyhow::Result<()> {
    // Detect if we are building with CUDA feature or environment
    println!("cargo:rerun-if-changed=src/kernels/bit_op.cu");
    println!("cargo:rerun-if-changed=build.rs");

    let cuda_file = "src/kernels/bit_op.cu";
    let ptx_file = "bit_op.ptx";

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let output_ptx = out_dir.join(ptx_file);

    // Attempt to find NVCC
    let nvcc = match env::var("CUDA_HOME") {
        Ok(home) => PathBuf::from(home).join("bin/nvcc"),
        Err(_) => PathBuf::from("nvcc"),
    };

    // Check if nvcc works
    let status = Command::new(&nvcc).arg("--version").output();
    if status.is_err() {
        println!(
            "cargo:warning=NVCC not found. Skipping CUDA kernel compilation. Using dummy PTX."
        );
        std::fs::write(&output_ptx, "")?;
        return Ok(());
    }

    let output = Command::new(&nvcc)
        .arg("-ptx")
        .arg("-arch=compute_80") // Target Ampere (RTX 30 series+) or adjust
        .arg("-code=sm_80")
        .arg(cuda_file)
        .arg("-o")
        .arg(&output_ptx)
        .output()?;

    if !output.status.success() {
        let err_str = String::from_utf8_lossy(&output.stderr);
        if err_str.contains("cl.exe") {
            println!("cargo:warning=NVCC could not find cl.exe (MSVC). CUDA kernels skipped (Non-fatal).");
        } else {
            println!(
                "cargo:warning=Failed to compile CUDA kernel. Using dummy PTX. Error: {}",
                err_str
            );
        }
        std::fs::write(&output_ptx, "")?;
    } else {
        println!(
            "cargo:warning=Successfully compiled CUDA kernel to {:?}",
            output_ptx
        );
    }

    Ok(())
}
