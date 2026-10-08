import torch
import triton
import triton.language as tl

# Define the block size and vector width
BLOCK_SIZE = 128
VECTOR_WIDTH = 4

@triton.jit
def _rmsnorm(x_ptr, w_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    
    # Load data with vectorized loads
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute the mean of squares using a more efficient reduction
    mean_squares = tl.zeros([VECTOR_WIDTH], dtype=tl.float32)
    for i in range(0, BLOCK, VECTOR_WIDTH):
        x_vec = tl.load(x_ptr + row * n_cols + i, mask=tl.arange(0, VECTOR_WIDTH) < (BLOCK - i), other=0.0).to(tl.float32)
        mean_squares += x_vec * x_vec
    mean_squares = tl.sum(mean_squares, axis=0) / n_cols
    
    # Compute the rsqrt
    rsqrt_mean_squares = tl.rsqrt(mean_squares + 1e-6)
    
    # Normalize and scale
    normalized_x = x * rsqrt_mean_squares
    scaled_x = normalized_x * w
    
    # Store the result
    tl.store(out_ptr + row * n_cols + cols, scaled_x.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, w):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    _rmsnorm[(n_rows,)](x, w, out, n_cols, BLOCK=BLOCK_SIZE)
    return out
