import torch
import triton
import triton.language as tl

@triton.jit
def _layernorm(x_ptr, w_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_elements
    x = tl.load(x_ptr + row * n_elements + cols, mask=mask, other=0.0).to(tl.float32)
    
    mean = tl.sum(x, axis=0) / n_elements
    x_centered = x - mean
    
    variance = tl.sum(x_centered * x_centered, axis=0) / n_elements
    rstd = tl.rsqrt(variance + 1e-5)
    
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    out = x_centered * rstd * w + b
    tl.store(out_ptr + row * n_elements + cols, out.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, w, b):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_elements = x.shape
    _layernorm[(n_rows,)](x, w, b, out, n_elements, BLOCK=triton.next_power_of_2(n_elements))
    return out
