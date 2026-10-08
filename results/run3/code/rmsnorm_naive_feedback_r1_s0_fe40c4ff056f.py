import torch
import triton
import triton.language as tl

# Define a larger block size for better utilization
BLOCK_SIZE = 128

@triton.jit
def _rmsnorm(x_ptr, w_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    # Each thread processes a row
    row = tl.program_id(0)
    # Each thread processes a block of columns
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    
    # Load x and w into registers
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute the mean of squares using shared memory
    smem = tl.zeros((BLOCK,), dtype=tl.float32)
    smem[cols] = x * x
    tl.block_sync()
    
    # Reduce the mean of squares
    for i in range(BLOCK // 2):
        if cols < BLOCK - i:
            smem[cols] += smem[cols + i]
    tl.block_sync()
    
    # Compute the rsqrt
    mean_squares = smem[0] / n_cols
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
