import torch
import triton
import triton.language as tl

# Define the block size
BLOCK_SIZE = 128

@triton.jit
def _rmsnorm(x_ptr, w_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    # Each row is processed by one program instance
    row = tl.program_id(0)
    # Load a chunk of the row into shared memory
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute the mean of squares using shared memory
    # Initialize shared memory
    sm = tl.zeros((BLOCK,), dtype=tl.float32)
    # Load data into shared memory
    sm[cols] = x * x
    # Synchronize threads to ensure all data is loaded
    tl.barrier()
    
    # Perform reduction in shared memory
    for stride in BLOCK // 2:
        if cols < stride:
            sm[cols] += sm[cols + stride]
        tl.barrier()
    
    # Compute the mean of squares
    mean_squares = sm[0] / n_cols
    
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
    # Launch the kernel with the appropriate grid and block size
    _rmsnorm[(n_rows,)](x, w, out, n_cols, BLOCK=BLOCK_SIZE)
    return out
