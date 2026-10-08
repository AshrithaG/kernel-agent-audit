import torch
import triton
import triton.language as tl

# Define the block size
BLOCK_SIZE = 128

@triton.jit
def _rmsnorm(x_ptr, w_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    # Define the number of elements per block
    ELEMS_PER_BLOCK = BLOCK // 4  # Using 4-element vectors (float32x4)
    
    # Get the row index
    row = tl.program_id(0)
    
    # Create a range of column indices
    cols = tl.arange(0, ELEMS_PER_BLOCK)
    
    # Create a mask to ensure we don't read out of bounds
    mask = cols < n_cols // 4
    
    # Load the input data into vectors
    x_vec = tl.load(x_ptr + row * n_cols + cols * 4, mask=mask, other=0.0).to(tl.float32)
    w_vec = tl.load(w_ptr + cols * 4, mask=mask, other=0.0).to(tl.float32)
    
    # Compute the mean of squares using parallel reduction
    mean_squares = tl.zeros([ELEMS_PER_BLOCK], dtype=tl.float32)
    for i in range(4):  # Unroll the loop for better performance
        x_elem = x_vec[i::4]
        mean_squares += x_elem * x_elem
    
    # Reduce the mean_squares vector to a single value
    mean_squares = tl.sum(mean_squares, axis=0) / n_cols
    
    # Compute the rsqrt
    rsqrt_mean_squares = tl.rsqrt(mean_squares + 1e-6)
    
    # Normalize and scale
    normalized_x = x_vec * rsqrt_mean_squares
    scaled_x = normalized_x * w_vec
    
    # Store the result
    tl.store(out_ptr + row * n_cols + cols * 4, scaled_x.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, w):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    _rmsnorm[(n_rows,)](x, w, out, n_cols, BLOCK=BLOCK_SIZE)
    return out
