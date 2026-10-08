import torch
import triton
import triton.language as tl

# Define a wider data type for vectorization
VECTORIZED_SIZE = 8

@triton.jit
def _rmsnorm(x_ptr, w_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK * VECTORIZED_SIZE)
    mask = cols < n_cols
    
    # Load data into vectors
    x_vec = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    w_vec = tl.load(w_ptr + cols // VECTORIZED_SIZE, mask=mask, other=0.0).to(tl.float32)
    
    # Compute the mean of squares
    mean_squares = tl.zeros([VECTORIZED_SIZE], dtype=tl.float32)
    for i in range(VECTORIZED_SIZE):
        mean_squares += x_vec[i::VECTORIZED_SIZE] * x_vec[i::VECTORIZED_SIZE]
    mean_squares /= n_cols
    
    # Compute the rsqrt
    rsqrt_mean_squares = tl.rsqrt(mean_squares + 1e-6)
    
    # Normalize and scale
    normalized_x = x_vec * rsqrt_mean_squares[cols // VECTORIZED_SIZE]
    scaled_x = normalized_x * w_vec
    
    # Store the result
    tl.store(out_ptr + row * n_cols + cols, scaled_x.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, w):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols // VECTORIZED_SIZE)
    _rmsnorm[(n_rows,)](x, w, out, n_cols, BLOCK=BLOCK)
    return out
