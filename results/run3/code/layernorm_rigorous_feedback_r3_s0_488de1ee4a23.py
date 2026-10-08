import torch
import triton
import triton.language as tl

@triton.jit
def _layernorm_optimized(x_ptr, w_ptr, b_ptr, out_ptr, n_features, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_features
    x = tl.load(x_ptr + row * n_features + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Shared memory for intermediate results
    smem_mean = tl.zeros(BLOCK, dtype=tl.float32)
    smem_sum_x = tl.zeros(BLOCK, dtype=tl.float32)
    smem_sum_x_sq = tl.zeros(BLOCK, dtype=tl.float32)
    
    # Compute mean and variance in a single pass
    smem_sum_x += x
    smem_sum_x_sq += x * x
    
    # Reduce to get the final mean and variance
    tl.block_reduce(smem_sum_x, lambda a, b: a + b)
    tl.block_reduce(smem_sum_x_sq, lambda a, b: a + b)
    
    mean = smem_sum_x / n_features
    var = (smem_sum_x_sq / n_features) - (mean * mean)
    rstd = tl.rsqrt(var + 1e-5)
    
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (x - mean) * rstd * w + b
    tl.store(out_ptr + row * n_features + cols, out.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, w, b):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_features = x.shape
    BLOCK = triton.next_power_of_2(n_features)
    if BLOCK > 1024:
        BLOCK = 1024
    _layernorm_optimized[(n_rows,)](x, w, b, out, n_features, BLOCK=BLOCK)
    return out
