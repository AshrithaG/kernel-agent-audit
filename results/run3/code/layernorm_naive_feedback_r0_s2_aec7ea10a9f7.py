import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import erf

@triton.jit
def _layernorm(x_ptr, w_ptr, b_ptr, out_ptr, n_features, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_features
    
    # Load data
    x = tl.load(x_ptr + row * n_features + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean
    mean = tl.sum(x, axis=1) / n_features
    mean = tl.broadcast_to(mean, (BLOCK,))
    
    # Compute variance
    var = tl.sum((x - mean) ** 2, axis=1) / n_features
    var = tl.broadcast_to(var, (BLOCK,))
    
    # Add epsilon to variance
    var = var + 1e-5
    
    # Compute inverse square root of variance
    rstd = tl.rsqrt(var)
    
    # Normalize
    x_normalized = (x - mean) * rstd
    
    # Load weight and bias
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Scale and shift
    y = x_normalized * w + b
    
    # Store result
    tl.store(out_ptr + row * n_features + cols, y.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, w, b):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_features = x.shape
    BLOCK = triton.next_power_of_2(n_features)
    _layernorm[(n_rows,)](x, w, b, out, n_features, BLOCK=BLOCK)
    return out
