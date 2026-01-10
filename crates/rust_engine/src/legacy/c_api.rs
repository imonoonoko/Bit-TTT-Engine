use super::ttt_layer::TTTLayer;
use ndarray::ArrayView2;
use std::slice;

/// Error Codes
#[repr(C)]
pub enum BitTTTError {
    Ok = 0,
    NullPointer = 1,
    DimensionMismatch = 2,
    Panic = 99,
}

/// Create a new TTTModel
/// Returns a raw pointer to the TTTLayer object.
#[no_mangle]
pub extern "C" fn ttt_create(hidden_dim: usize, inner_lr: f32) -> *mut TTTLayer {
    let model = TTTLayer::new(hidden_dim, inner_lr);
    Box::into_raw(Box::new(model))
}

/// Destroy the TTTModel to free memory.
/// Must be called once done with the pointer.
///
/// # Safety
/// - `ptr` must be a valid pointer obtained from `ttt_create`.
/// - Double-freeing (calling this twice on same ptr) causes Undefined Behavior.
#[no_mangle]
pub unsafe extern "C" fn ttt_destroy(ptr: *mut TTTLayer) {
    if !ptr.is_null() {
        // Drop the box, freeing memory
        let _ = Box::from_raw(ptr);
    }
}

/// Forward pass for a single sequence.
/// Returns error code (0 = Ok).
///
/// # Safety
/// - `ptr` must be a valid pointer to TTTLayer created by ttt_create.
/// - `input_ptr` must point to an array of size `seq_len * hidden_dim`.
/// - `output_ptr` must point to a buffer of size `seq_len * hidden_dim`.
#[no_mangle]
pub unsafe extern "C" fn ttt_forward(
    ptr: *mut TTTLayer,
    input_ptr: *const f32,
    seq_len: usize,
    output_ptr: *mut f32,
) -> i32 {
    // # Safety
    // This function acts as a raw FFI entry point.
    // The caller *MUST* guarantee:
    // 1. `ptr` is a valid pointer returned by `ttt_create` and has not been destroyed.
    // 2. `input_ptr` points to a valid memory region of at least `seq_len * hidden_dim` f32 elements.
    // 3. `output_ptr` points to a mutable memory region of at least `seq_len * hidden_dim` f32 elements.
    // 4. `seq_len` matches the actual data length provided.
    //
    // Internal checks for NULL are performed, but invalid non-null pointers will cause Undefined Behavior.

    // Validate non-null inputs immediately
    if ptr.is_null() || input_ptr.is_null() || output_ptr.is_null() {
        return BitTTTError::NullPointer as i32;
    }

    // DEBUG: Basic sanity check
    debug_assert!(seq_len > 0, "Sequence length must be positive");

    let model = &*ptr; // SAFETY: ptr is checked non-null. Caller guarantees validity.
    let dim = model.hidden_dim;

    // SAFETY: Prevent integer overflow when calculating buffer size
    let total_len = match seq_len.checked_mul(dim) {
        Some(len) => len,
        None => return BitTTTError::DimensionMismatch as i32, // Overflow = invalid dimensions
    };

    // Create view from raw C pointers
    // SAFETY: Validity of pointers and length is guaranteed by the caller (C-ABI contract).
    let input_slice = slice::from_raw_parts(input_ptr, total_len);
    let output_slice = slice::from_raw_parts_mut(output_ptr, total_len);

    // Convert to ndarray (View) -> Zero-Copy!
    let input_view = match ArrayView2::from_shape((seq_len, dim), input_slice) {
        Ok(view) => view,
        Err(_) => return BitTTTError::DimensionMismatch as i32,
    };

    let result = model.forward_sequence(&input_view);

    // Copy result back to output buffer
    for (i, &val) in result.iter().enumerate() {
        if i < total_len {
            output_slice[i] = val;
        }
    }

    BitTTTError::Ok as i32
}
