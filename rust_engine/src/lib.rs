use ndarray::{Array1, Array2};
use std::slice;

pub mod bit_linear;
pub mod ttt_layer;

pub use bit_linear::BitLinear;
pub use ttt_layer::TTTLayer;

/// C-ABI Interface

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

/// Forward pass for a single sequence (simplified).
/// Takes input array pointer, length, and fills output array pointer.
///
/// input_ptr: Pointer to float array of size (seq_len * hidden_dim)
/// output_ptr: Pointer to buffer of size (seq_len * hidden_dim)
#[no_mangle]
pub unsafe extern "C" fn ttt_forward(
    ptr: *mut TTTLayer,
    input_ptr: *const f32,
    seq_len: usize,
    output_ptr: *mut f32,
) {
    let model = &*ptr; // Borrow the model
    let dim = model.hidden_dim;
    let total_len = seq_len * dim;

    // Create view from raw C pointers
    let input_slice = slice::from_raw_parts(input_ptr, total_len);
    let output_slice = slice::from_raw_parts_mut(output_ptr, total_len);

    // Convert to ndarray (View)
    // Note: Array2::from_shape_vec would copy, usually.
    // Here we construct a view.
    // Convert to ndarray (View)
    // Safety Check: Ensure dimensions match
    let input_vec = input_slice.to_vec();
    let input_array = match Array2::from_shape_vec((seq_len, dim), input_vec) {
        Ok(arr) => arr,
        Err(e) => {
            eprintln!("Error creating ndarray from shape: {}", e);
            return;
        }
    };

    let result = model.forward_sequence(&input_array);

    // Copy result back to output buffer
    // result is Array2<f32>.
    // Assuming row-major standard layout.
    for (i, &val) in result.iter().enumerate() {
        if i < total_len {
            output_slice[i] = val;
        }
    }
}
