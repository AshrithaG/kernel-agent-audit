import torch
import triton
import triton.language as tl

@triton.jit
def _layernorm(x_ptr, w_ptr, b_ptr, out_ptr, n_features, BLOCK: tl.constexpr):
    # Each thread block processes a row
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_features
    
    # Load data into registers
    x = tl.load(x_ptr + row * n_features + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / n_features
    mean = tl.broadcast_to(mean, (BLOCK,))
    x_centered = x - mean
    
    # Compute variance
    var = tl.sum(x_centered * x_centered, axis=0) / n_features
    rstd = tl.rsqrt(var + 1e-5)
    
    # Load weights and biases
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
    _layernorm[(n_rows,)](x, w, b, out, n_features, BLOCK=BLOCK)
    return out
