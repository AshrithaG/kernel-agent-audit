import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm(x_ptr, w_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute the mean of squares
    mean_squares = tl.sum(x * x, axis=0) / n_cols
    
    # Compute the RMSNorm
    rmsnorm = x * tl.rsqrt(mean_squares + 1e-6)
    
    # Apply the weight
    result = rmsnorm * w
    
    # Store the result
    tl.store(out_ptr + row * n_cols + cols, result.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, w):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    _rmsnorm[(n_rows,)](x, w, out, n_cols, BLOCK=triton.next_power_of_2(n_cols))
    return out
