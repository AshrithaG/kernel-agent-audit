import torch
import triton
import triton.language as tl

@triton.jit
def _layernorm(x_ptr, w_ptr, b_ptr, out_ptr, n_features, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_features
    x = tl.load(x_ptr + row * n_features + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean
    mean = tl.sum(x, axis=1) / n_features
    mean = tl.broadcast_to(mean, (BLOCK,))
    
    # Compute variance
    var = tl.sum((x - mean) ** 2, axis=1) / n_features
    var = tl.broadcast_to(var, (BLOCK,))
    
    # Compute normalized values
    epsilon = 1e-5
    inv_std = tl.rsqrt(var + epsilon)
    normalized = (x - mean) * inv_std
    
    # Apply weight and bias
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    result = normalized * w + b
    
    # Store result
    tl.store(out_ptr + row * n_features + cols, result.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, w, b):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_features = x.shape
    _layernorm[(n_rows,)](x, w, b, out, n_features, BLOCK=triton.next_power_of_2(n_features))
    return out
