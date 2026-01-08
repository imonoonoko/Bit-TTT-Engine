use crate::TTTLayer;
use ndarray::Array2;
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
#[no_mangle]
pub unsafe extern "C" fn ttt_destroy(ptr: *mut TTTLayer) {
    if !ptr.is_null() {
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
    // SAFETY: We verify ptr is not null immediately.
    if ptr.is_null() || input_ptr.is_null() || output_ptr.is_null() {
        return BitTTTError::NullPointer as i32;
    }

    let model = &*ptr; // SAFETY: ptr is checked non-null. Caller guarantees validity.
    let dim = model.hidden_dim;
    let total_len = seq_len * dim;

    // Create view from raw C pointers
    // SAFETY: Validity of pointers and length is guaranteed by the caller (C-ABI contract).
    let input_slice = slice::from_raw_parts(input_ptr, total_len);
    let output_slice = slice::from_raw_parts_mut(output_ptr, total_len);

    // Convert to ndarray (View)
    // Safety Check: We clone to vec for safety and simplicity here.
    let input_vec = input_slice.to_vec();
    let input_array = match Array2::from_shape_vec((seq_len, dim), input_vec) {
        Ok(arr) => arr,
        Err(_) => return BitTTTError::DimensionMismatch as i32,
    };

    let result = model.forward_sequence(&input_array);

    // Copy result back to output buffer
    for (i, &val) in result.iter().enumerate() {
        if i < total_len {
            output_slice[i] = val;
        }
    }

    BitTTTError::Ok as i32
}
