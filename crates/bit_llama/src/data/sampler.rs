use anyhow::Result;
use rayon::prelude::*;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::sync_channel;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::thread;

pub struct ParallelSampler;

impl ParallelSampler {
    pub fn sample(
        files: Vec<String>,
        output_path: PathBuf,
        limit_mb: usize,
    ) -> Result<Vec<String>> {
        println!(
            "⚡ Optimization: Parallel Sampling first {} MB of data...",
            limit_mb
        );

        let limit_bytes = limit_mb * 1_024 * 1_024;

        let (tx, rx) = sync_channel::<Vec<u8>>(4); // Backpressure: Max 4 chunks in flight
        let total_written = Arc::new(AtomicUsize::new(0));
        let total_written_clone = total_written.clone();

        // 1. Writer Thread (Consumer)
        let sample_path_clone = output_path.clone();
        let writer_handle = thread::spawn(move || -> Result<usize> {
            let out_sample = std::fs::File::create(&sample_path_clone)?;
            let mut writer = std::io::BufWriter::with_capacity(4 * 1024 * 1024, out_sample); // 4MB Buffer
            let mut bytes_count = 0;

            for chunk in rx {
                writer.write_all(&chunk)?;
                bytes_count += chunk.len();
                total_written_clone.fetch_add(chunk.len(), Ordering::Relaxed);
            }
            writer.flush()?;
            Ok(bytes_count)
        });

        // 2. Reader Threads (Producers)
        let global_bytes = Arc::new(AtomicUsize::new(0));

        files
            .par_iter()
            .try_for_each_with(tx, |s, path_str| -> Result<()> {
                // Early exit check (Relaxed is fine for rough limit)
                if global_bytes.load(Ordering::Relaxed) >= limit_bytes {
                    return Ok(());
                }

                let path = Path::new(path_str);
                if let Ok(f) = std::fs::File::open(path) {
                    let mut reader = std::io::BufReader::with_capacity(1024 * 1024, f);
                    let mut buffer = [0u8; 1024 * 1024]; // 1MB Chunk

                    loop {
                        // Check global limit inside loop for large files (Atomic Relaxed is cheap)
                        if global_bytes.load(Ordering::Relaxed) >= limit_bytes {
                            break;
                        }

                        match reader.read(&mut buffer) {
                            Ok(0) => break, // EOF
                            Ok(n) => {
                                if s.send(buffer[..n].to_vec()).is_err() {
                                    break; // Channel closed
                                }
                                global_bytes.fetch_add(n, Ordering::Relaxed);
                            }
                            Err(_) => break, // Read error
                        }
                    }
                }
                Ok(())
            })?;

        // Channels are dropped here, Writer thread will finish when empty

        let written = writer_handle
            .join()
            .map_err(|_| anyhow::anyhow!("Writer thread panicked"))??;

        println!(
            "   Sample created: {:?} ({} MB)",
            output_path,
            written / 1_024 / 1_024
        );
        Ok(vec![output_path.to_string_lossy().to_string()])
    }
}
