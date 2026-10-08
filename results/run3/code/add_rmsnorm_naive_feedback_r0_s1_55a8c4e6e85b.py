import torch
import triton
import triton.language as tl

@triton.jit
def _add_rmsnorm_kernel(x_ptr, r_ptr, w_ptr, out_ptr, n_elements, n_features, BLOCK: tl.constexpr):
    # Each block processes one row
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_features
    
    # Load data
    x = tl.load(x_ptr + row * n_features + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * n_features + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr, mask=mask, other=0.0).to(tl.float32)
    
    # Compute sum of squares
    sum_of_squares = tl.sum(x * x, axis=0)
    
    # Compute RMSNorm
    rms = tl.rsqrt(sum_of_squares + 1e-6)
    
    # Compute output
    h = (x + r) * rms * w
    
    # Store output
    tl.store(out_ptr + row * n_features + cols, h.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, r, w):
    x = x.contiguous()
    r = r.contiguous()
    w = w.contiguous()
    out = torch.empty_like(x)
    n_rows, n_features = x.shape
    _add_rmsnorm_kernel[(n_rows,)](x, r, w, out, n_rows, n_features, BLOCK=triton.next_power_of_2(n_features))
    return out
