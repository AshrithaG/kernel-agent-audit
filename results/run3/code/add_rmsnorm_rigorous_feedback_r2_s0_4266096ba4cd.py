import torch
import triton
import triton.language as tl

# Define a block size that is a power of two
BLOCK_SIZE = 128

@triton.jit
def _add_rmsnorm(x_ptr, r_ptr, w_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    
    # Load data
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Add residual
    h = x + r
    
    # Compute RMSNorm
    sum_squares = tl.sum(h * h, axis=0)
    rms = tl.sqrt(sum_squares / n_cols + 1e-6)
    inv_rms = tl.rsqrt(rms)
    
    # Load weight
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Normalize and scale
    normalized_h = h * inv_rms
    scaled_h = normalized_h * w
    
    # Store result
    tl.store(out_ptr + row * n_cols + cols, scaled_h.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, r, w):
    x = x.contiguous()
    r = r.contiguous()
    w = w.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    _add_rmsnorm[(n_rows,)](x, r, w, out, n_cols, BLOCK=BLOCK_SIZE)
    return out
