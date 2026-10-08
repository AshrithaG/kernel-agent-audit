import torch
import triton
import triton.language as tl

@triton.jit
def _layernorm_optimized(x_ptr, w_ptr, b_ptr, out_ptr, n_features, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_features
    x = tl.load(x_ptr + row * n_features + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / n_features
    mean = tl.broadcast_to(mean, (BLOCK,))
    
    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n_features
    
    # Compute reciprocal square root of variance
    rstd = tl.rsqrt(var + 1e-5)
    
    # Load weight and bias
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute output
    out = x_centered * rstd * w + b
    
    # Store output
    tl.store(out_ptr + row * n_features + cols, out.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, w, b):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_features = x.shape
    BLOCK = triton.next_power_of_2(n_features)
    grid = (n_rows,)
    _layernorm_optimized[grid](x, w, b, out, n_features, BLOCK=BLOCK)
    return out
