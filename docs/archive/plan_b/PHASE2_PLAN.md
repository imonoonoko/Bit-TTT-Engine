# Phase 2 Implementation Plan

**Objective**: Enhance system reliability (data integrity) and usability (input handling).

## 1. Dependencies
- [ ] Add `fs2 = "0.4"` to `crates/rust_engine/Cargo.toml`.
- [ ] Add `fs2 = "0.4"` to `crates/bit_llama/Cargo.toml`.
- [ ] Add `memmap2 = "0.9"` to `crates/rust_engine/Cargo.toml`.

## 2. UX Improvements (Inference Playground)
- [ ] Modify `crates/bit_llama/src/gui/tabs/inference.rs`:
    - Update `send_message` trigger: `ui.input(|i| i.modifiers.command && i.key_pressed(Key::Enter))`.
    - Update Placeholder text: `"Type a message... (Ctrl+Enter to send)"`.
    - Remove old `Shift+Enter` logic if confusing, or keep as alternative. (Suggest: keep Shift+Enter for newline, Enter for newline, Ctrl+Enter for send).

## 3. Data Integrity (File Locking)
### 3.1. Utility
- [ ] Create `crates/rust_engine/src/utils/file_lock.rs` (Optional) or use inline.
    - Helper trait/function to handle `lock_exclusive` and `lock_shared` with retries or timeouts.

### 3.2. Training (Exclusive Lock)
- [ ] Modify `crates/bit_llama/src/train/training_loop.rs` (`save_model` function):
    - Before writing `.safetensors`:
        ```rust
        let file = fs::File::create(&path)?;
        file.lock_exclusive()?;
        // write...
        file.unlock()?; // Explicit or implicit on drop
        ```

### 3.3. Inference (Shared Lock & Mmap)
- [ ] Modify `crates/rust_engine/src/model/llama.rs` (`new_with_weights` / `load_tensors`):
    - **Strategy**: Avoid TOCTOU race condition by keeping the lock during load.
    1. Open File: `let file = fs::File::open(&path)?;`
    2. Lock Shared: `file.lock_shared()?;`
    3. Mmap: `let mmap = unsafe { MmapOptions::new().map(&file)? };`
    4. Deserialize: `let tensors = SafeTensors::deserialize(&mmap)?;`
    5. *Note*: We need to ensure `mmap` stays alive as long as `tensors` are used. This might require refactoring `Llama` to own the `Mmap` object if it doesn't already own the data source.
    - Alternatively, simpler approach for now:
        - Lock Shared.
        - Read file to memory (`Vec<u8>`).
        - Unlock.
        - `SafeTensors::deserialize` from Vec.
        - This avoids lifetime issues with Mmap but uses more RAM. Given "Bit-Llama" might be small, this could be acceptable.
        - **Decision**: Start with Mmap strategy if structure allows, else Vec strategy. The User suggested Mmap. I will trust the user and attempt Mmap, but might need to adjust `Llama` struct to hold `Mmap`.

## 4. Verification
- [ ] Build & Run.
- [ ] Test Chat with `Ctrl+Enter` - ensure IME Enter doesn't send.
- [ ] (Optional) Simulate long save and try to load (Manual verification).
