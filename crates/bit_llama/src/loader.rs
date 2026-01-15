use anyhow::Result;
use candle_core::{Device, Tensor};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

pub struct BitLoader {
    _file: File,                 // Keep file handle alive
    mmap: Mmap,                  // Memory map
    pub mask_mmap: Option<Mmap>, // Optional mask file
    pub data_len: usize,         // count of tokens
    pub cursor: usize,
    pub is_u32: bool,    // Flag for 32-bit tokens
    pub loop_data: bool, // If true, reset cursor on EOF
}

impl BitLoader {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let is_u32 = path
            .extension()
            .is_some_and(|ext| ext == "u32" || path.to_string_lossy().ends_with(".u32.bin"));

        // Check for mask file (same name but .mask extension)
        let mask_path = path.with_extension("mask");
        let mask_mmap = if mask_path.exists() {
            let mask_file = File::open(&mask_path)?;
            let mm = unsafe { Mmap::map(&mask_file)? };
            tracing::info!("BitLoader: Found mask file: {:?}", mask_path);

            // Verify size alignment
            let expected_len = if is_u32 {
                mmap.len() / 4
            } else {
                mmap.len() / 2
            };
            if mm.len() != expected_len {
                tracing::warn!(
                    "BitLoader: Mask size mismatch! Data tokens: {}, Mask bytes: {}",
                    expected_len,
                    mm.len()
                );
            }
            Some(mm)
        } else {
            None
        };

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
            mask_mmap,
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
        let (inp, tgt, _) = self.next_batch_masked(batch_size, len, device)?;
        Ok((inp, tgt))
    }

    pub fn next_batch_masked(
        &mut self,
        batch_size: usize,
        len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let mut inputs = Vec::with_capacity(batch_size * len);
        let mut targets = Vec::with_capacity(batch_size * len);
        let mut masks = Vec::with_capacity(batch_size * len);

        let elem_size = if self.is_u32 { 4 } else { 2 };
        let has_mask = self.mask_mmap.is_some();

        for _ in 0..batch_size {
            if self.cursor + len + 1 >= self.data_len {
                if self.loop_data {
                    self.cursor = 0; // Reset (looping)
                } else {
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

            if let Some(ref mmap) = self.mask_mmap {
                // Mask matches token index directly (byte per token)
                let start_idx = self.cursor;
                // We need masks for the TARGETS (positions 1..len+1 relative to cursor)
                // usually mask is aligned with input or target?
                // Our PrepareInstruct generated mask for every token.
                // If input[i] predicts input[i+1], the loss is on input[i+1].
                // So we should take mask[1..len+1]?
                // Or is the mask associated with the position we are predicting AT?
                // If ID[0] is user instruction, ID[1] is user instruction...
                // We predict ID[1] from ID[0]. If ID[1] is instruction, we want to IGNORE loss on ID[1].
                // So mask[1] tells us if we should learn ID[1].
                // Yes, slice [1..len+1].

                let chunk_mask = &mmap[start_idx..start_idx + len + 1];
                // We extend using slice [1..] corresponding to targets
                // Convert u8 to f32 (0.0 or 1.0) for easy multiplication?
                // Or keep as u8/u32? Tensor doesn't support u8 well in Candle 0.3?
                // Let's us u32 or f32. u8 is supported in newer candle.
                // Let's cast to f32 0.0/1.0 immediately for safety.
                // PrepareInstruct: 0=Learn, 1=Ignore.
                // We typically multiply Loss by Weight.
                // So we want Weight=1 for Learn, Weight=0 for Ignore.
                // So we need to invert: new_mask = 1.0 - old_mask.
                // Or simple: 0 -> 1.0, 1 -> 0.0.

                for &m in &chunk_mask[1..len + 1] {
                    masks.push(if m == 0 { 1.0f32 } else { 0.0f32 });
                }
            }

            self.cursor += len;
        }

        let inp_tensor = Tensor::from_slice(&inputs, (batch_size, len), device)?;
        let tgt_tensor = Tensor::from_slice(&targets, (batch_size, len), device)?;

        let mask_tensor = if has_mask {
            Some(Tensor::from_slice(&masks, (batch_size, len), device)?)
        } else {
            None
        };

        Ok((inp_tensor, tgt_tensor, mask_tensor))
    }
}
