import torch
import triton
import triton.language as tl
from triton.language import extra

@triton.jit
def _add_rmsnorm_kernel(x_ptr, r_ptr, w_ptr, out_ptr, n_elements, n_features, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_features
    
    # Load x, r, and w
    x = tl.load(x_ptr + row * n_features + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * n_features + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Add residual
    h = x + r
    
    # Compute mean square
    mean_square = tl.sum(h * h, axis=0) / n_elements
    
    # Compute rsqrt
    rsqrt_mean_square = tl.rsqrt(mean_square + 1e-6)
    
    # Normalize and scale
    normalized_h = h * rsqrt_mean_square
    scaled_h = normalized_h * w
    
    # Store result
    tl.store(out_ptr + row * n_features + cols, scaled_h.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, r, w):
    x = x.contiguous()
    r = r.contiguous()
    w = w.contiguous()
    out = torch.empty_like(x)
    
    n_rows, n_features = x.shape
    _add_rmsnorm_kernel[(n_rows,)](x, r, w, out, n_rows, n_features, BLOCK=triton.next_power_of_2(n_features))
    return out
