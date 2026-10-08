import torch
import triton
import triton.language as tl

@triton.jit
def _layernorm(x_ptr, w_ptr, b_ptr, out_ptr, mean_ptr, var_ptr, n_elements, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_elements
    
    # Load data
    x = tl.load(x_ptr + row * n_elements + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / n_elements
    tl.store(mean_ptr + row, mean, mask=mask)
    
    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n_elements
    tl.store(var_ptr + row, var, mask=mask)
    
    # Compute output
    epsilon = 1e-5
    inv_var = tl.rsqrt(var + epsilon)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normalized_x = x_centered * inv_var
    output = normalized_x * w + b
    tl.store(out_ptr + row * n_elements + cols, output, mask=mask)

def kernel(x, w, b):
    x = x.contiguous()
    out = torch.empty_like(x)
    mean = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
    var = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
    n_elements = x.shape[1]
    BLOCK = triton.next_power_of_2(n_elements)
    _layernorm[(x.shape[0],)](x, w, b, out, mean, var, n_elements, BLOCK=BLOCK)
    return out
