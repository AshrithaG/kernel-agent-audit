import torch
import triton
import triton.language as tl

@triton.jit
def _layernorm(x_ptr, w_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # Load data
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_elements
    x = tl.load(x_ptr + pid * n_elements + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / n_elements
    mean = tl.broadcast_to(mean, (BLOCK,))
    
    # Compute variance
    var = tl.sum((x - mean) * (x - mean), axis=0) / n_elements
    var = tl.broadcast_to(var, (BLOCK,))
    
    # Compute inverse sqrt of variance
    inv_sqrt_var = tl.rsqrt(var + 1e-5)
    
    # Normalize
    normalized = (x - mean) * inv_sqrt_var
    
    # Scale and shift
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    result = normalized * w + b
    
    # Store result
    tl.store(out_ptr + pid * n_elements + cols, result.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, w, b):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_elements = x.shape
    _layernorm[(n_rows,)](x, w, b, out, n_elements, BLOCK=triton.next_power_of_2(n_elements))
    return out
