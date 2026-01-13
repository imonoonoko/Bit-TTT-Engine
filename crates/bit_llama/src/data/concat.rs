use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::sync::mpsc::Sender;
use std::thread;
use glob::glob;

pub struct Concatenator {
    pub input_pattern: String,
    pub output_path: PathBuf,
    pub cancel_flag: Arc<AtomicBool>,
    pub log_tx: Sender<String>,
}

impl Concatenator {
    pub fn new(
        input_pattern: String,
        output_path: PathBuf,
        cancel_flag: Arc<AtomicBool>,
        log_tx: Sender<String>,
    ) -> Self {
        Self {
            input_pattern,
            output_path,
            cancel_flag,
            log_tx,
        }
    }

    pub fn run(self) {
        let cancel_flag = self.cancel_flag.clone();
        let log_tx = self.log_tx.clone();
        let pattern = self.input_pattern.clone();
        let output_path = self.output_path.clone();

        thread::spawn(move || {
            let mut count = 0;
            let mut total_bytes = 0;

            // 4MB Buffer for NVMe Optimization
            const CHUNK_SIZE: usize = 4 * 1024 * 1024;

            match glob(&pattern) {
                Ok(paths) => {
                    match fs::File::create(&output_path) {
                        Ok(file) => {
                            let mut out_file = std::io::BufWriter::with_capacity(CHUNK_SIZE, file);

                            for entry in paths {
                                // Check Cancel
                                if cancel_flag.load(Ordering::Relaxed) {
                                    let _ = log_tx.send("🛑 Concatenation Cancelled by User.".to_string());
                                    let _ = log_tx.send("<<CONCAT_DONE>>".to_string());
                                    return;
                                }

                                if let Ok(path) = entry {
                                    if path.is_file() {
                                        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
                                        let valid_exts = ["txt", "md", "json", "jsonl"];
                                        if valid_exts.contains(&ext) {
                                            if let Ok(mut in_file) = fs::File::open(&path) {
                                                match std::io::copy(&mut in_file, &mut out_file) {
                                                    Ok(bytes) => {
                                                        total_bytes += bytes as usize;
                                                    }
                                                    Err(e) => {
                                                        let _ = log_tx.send(format!("Write error: {}", e));
                                                    }
                                                }
                                                let _ = out_file.write_all(b"\n");
                                                count += 1;

                                                if count % 100 == 0 {
                                                     let _ = log_tx.send(format!("Processed {} files...", count));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // Flush logic implies closure or explicit flush
                            if let Err(e) = out_file.flush() {
                                let _ = log_tx.send(format!("Flush error: {}", e));
                            }
                        }
                        Err(e) => {
                            let _ = log_tx.send(format!("❌ Failed to create output file: {}", e));
                        }
                    }
                },
                Err(e) => {
                    let _ = log_tx.send(format!("❌ Glob pattern error: {}", e));
                }
            }

            if count == 0 {
                 let _ = log_tx.send(format!("⚠️ No .txt or .md matches for '{}'", pattern));
            } else {
                 let _ = log_tx.send(format!(
                    "✅ Concatenated {} files in {:.2} MB.",
                    count,
                    total_bytes as f64 / 1_048_576.0
                ));
            }
             let _ = log_tx.send("<<CONCAT_DONE>>".to_string());
        });
    }
}
