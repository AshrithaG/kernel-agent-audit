import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute the square of x
    x_sq = x * x
    
    # Perform reduction to get the mean of squares
    sum_x_sq = tl.sum(x_sq, axis=0)
    mean_x_sq = sum_x_sq / n_cols
    
    # Add epsilon and compute the reciprocal square root
    epsilon = tl.constexpr(1e-6)
    inv_sqrt_mean_x_sq = libdevice.rsqrt(mean_x_sq + epsilon)
    
    # Normalize and scale
    normalized_x = x * inv_sqrt_mean_x_sq
    scaled_x = normalized_x * w
    
    # Store the result
    tl.store(out_ptr + row * n_cols + cols, scaled_x.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, w):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    _rmsnorm_kernel[(n_rows,)](x, w, out, n_cols, BLOCK=triton.next_power_of_2(n_cols))
    return out
