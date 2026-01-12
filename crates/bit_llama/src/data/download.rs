use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::path::Path;

/// Download a file from a URL to a local path with a progress bar.
pub fn download_file(url: &str, output_path: &Path) -> Result<()> {
    println!("📥 Downloading: {}", url);

    let resp = ureq::get(url).call().context("Failed to send request")?;

    let total_size: u64 = resp
        .header("Content-Length")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0u64);

    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
        .progress_chars("#>-"));

    let mut source = resp.into_reader();
    let mut dest = File::create(output_path).context("Failed to create output file")?;

    // Copy with progress
    // ureq reader doesn't automatically update progress bar, so we might need a wrapper or manual buffer loop
    // to update PB. For simplicity with standard io::copy, we lose progress updates unless we wrap.
    // Let's implement a simple buffer loop.

    let mut buffer = [0; 8192];
    let mut downloaded = 0;

    loop {
        let n = std::io::Read::read(&mut source, &mut buffer)?;
        if n == 0 {
            break;
        }
        std::io::Write::write_all(&mut dest, &buffer[..n])?;
        downloaded += n as u64;
        pb.set_position(downloaded);
    }

    pb.finish_with_message("Download complete");
    println!("✅ Saved to: {:?}", output_path);
    Ok(())
}

const DEFAULT_JAPANESE_CORPUS_URL: &str = "https://huggingface.co/datasets/izumi-lab/llm-japanese-dataset/resolve/main/data-cc-by-sa.jsonl";

/// Downloads the Wiki40b-Ja dataset (or similar).
/// Since Wiki40b is large and usually TFRecord, we might want a simpler JSONL or txt version for this project.
/// Using a HuggingFace generic raw file link or similar if available.
/// For now, we'll download a sample or a specific "tiny" japanese corpus if Wiki40b is too complex to parse raw.
///
/// Note: Real Wiki40b requires parsing. For Phase 14 MVP, we might use a pre-processed plain text URL
/// or just instructions.
///
/// Let's default to a placeholder or a reliable open text corpus URL (e.g. Aozora Bunko subset or similar).
/// Actually, `mc4` or `oscar` samples from HF are good.
pub fn download_wiki40b_ja_sample(output_dir: &Path) -> Result<std::path::PathBuf> {
    let url = DEFAULT_JAPANESE_CORPUS_URL;
    let output_path = output_dir.join("corpus_ja.jsonl");

    // Actually download
    download_file(url, &output_path)?;

    Ok(output_path)
}
