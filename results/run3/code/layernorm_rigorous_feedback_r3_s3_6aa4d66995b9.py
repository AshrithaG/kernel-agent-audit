import torch
import triton
import triton.language as tl

@triton.jit
def _layernorm(x_ptr, w_ptr, b_ptr, out_ptr, n_features, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_features
    x = tl.load(x_ptr + row * n_features + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean and variance in a single pass
    sum_x = tl.sum(x, axis=0)
    sum_x_sq = tl.sum(x * x, axis=0)
    mean = sum_x / n_features
    mean = tl.broadcast_to(mean, (BLOCK,))
    var = (sum_x_sq / n_features) - (mean * mean)
    rstd = tl.rsqrt(var + 1e-5)
    
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (x - mean) * rstd * w + b
    tl.store(out_ptr + row * n_features + cols, out.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, w, b):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_features = x.shape
    BLOCK = 1024  # Adjust this value based on your GPU's capabilities
    grid = (triton.cdiv(n_rows, BLOCK),)
    _layernorm[grid](x, w, b, out, n_features, BLOCK=BLOCK)
    return out
