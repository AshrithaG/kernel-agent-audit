import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_optimized(x_ptr, w_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    
    # Load data into shared memory
    smem = tl.zeros((BLOCK,), dtype=tl.float32)
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    smem[cols] = x * x
    
    # Parallel reduction in shared memory
    tl.block_range = BLOCK
    for offset in range(BLOCK // 2):
        if cols >= offset:
            smem[cols] += smem[cols + offset]
    tl.barrier()
    
    # Compute the mean of squares
    mean_squares = smem[0] / n_cols
    
    # Compute the rsqrt
    rsqrt_mean_squares = tl.rsqrt(mean_squares + 1e-6)
    
    # Normalize and scale
    normalized_x = x * rsqrt_mean_squares
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    scaled_x = normalized_x * w
    
    # Store the result
    tl.store(out_ptr + row * n_cols + cols, scaled_x.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, w):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    _rmsnorm_optimized[(n_rows,)](x, w, out, n_cols, BLOCK=BLOCK)
    return out
