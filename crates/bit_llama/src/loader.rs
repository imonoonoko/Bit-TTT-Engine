use anyhow::Result;
use candle_core::{Device, Tensor};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

pub struct BitLoader {
    _file: File,         // Keep file handle alive
    mmap: Mmap,          // Memory map
    pub data_len: usize, // count of tokens
    pub cursor: usize,
    pub is_u32: bool,    // Flag for 32-bit tokens
    pub loop_data: bool, // If true, reset cursor on EOF
}

impl BitLoader {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let is_u32 = path.extension().map_or(false, |ext| {
            ext == "u32" || path.to_string_lossy().ends_with(".u32.bin")
        });

        let data_len = if is_u32 {
            mmap.len() / 4
        } else {
            mmap.len() / 2
        };

        tracing::info!(
            "BitLoader: Loading {:?} (Model: {}, Tokens: {})",
            path,
            if is_u32 { "u32" } else { "u16" },
            data_len
        );

        Ok(Self {
            _file: file,
            mmap,
            data_len,
            cursor: 0,
            is_u32,
            loop_data: true, // Default to true (training mode)
        })
    }

    pub fn with_loop(mut self, loop_data: bool) -> Self {
        self.loop_data = loop_data;
        self
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    pub fn next_batch(
        &mut self,
        batch_size: usize,
        len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let mut inputs = Vec::with_capacity(batch_size * len);
        let mut targets = Vec::with_capacity(batch_size * len);

        let elem_size = if self.is_u32 { 4 } else { 2 };

        for _ in 0..batch_size {
            if self.cursor + len + 1 >= self.data_len {
                if self.loop_data {
                    self.cursor = 0; // Reset (looping)
                } else {
                    // Stop if not looping and end of data
                    // If we have partial batch, we might return it if handled,
                    // but here we just error or break logic.
                    // For evaluate, we typically want to handle 'End of Data' gracefully.
                    return Err(anyhow::anyhow!("End of Data"));
                }
            }

            let start = self.cursor * elem_size;
            let end = (self.cursor + len + 1) * elem_size;
            let chunk_raw = &self.mmap[start..end];

            let chunk_u32: Vec<u32> = if self.is_u32 {
                chunk_raw
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            } else {
                chunk_raw
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
                    .collect()
            };

            inputs.extend_from_slice(&chunk_u32[0..len]);
            targets.extend_from_slice(&chunk_u32[1..len + 1]);

            self.cursor += len;
        }

        let inp_tensor = Tensor::from_slice(&inputs, (batch_size, len), device)?;
        let tgt_tensor = Tensor::from_slice(&targets, (batch_size, len), device)?;

        Ok((inp_tensor, tgt_tensor))
    }
}
