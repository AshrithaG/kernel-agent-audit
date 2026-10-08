import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import erf

@triton.jit
def _layernorm(x_ptr, w_ptr, b_ptr, out_ptr, n_elements, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_elements
    x = tl.load(x_ptr + row * n_elements + cols, mask=mask, other=0.0).to(tl.float32)
    
    mean = tl.sum(x, axis=1) / n_elements
    mean_bcast = tl.broadcast_to(mean[:, None], (BLOCK,))
    x_centered = x - mean_bcast
    
    var = tl.sum(x_centered * x_centered, axis=1) / n_elements
    var_bcast = tl.broadcast_to(var[:, None], (BLOCK,))
    rstd = tl.rsqrt(var_bcast + eps)
    
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    normalized = x_centered * rstd
    scaled = normalized * w
    result = scaled + b
    
    tl.store(out_ptr + row * n_elements + cols, result.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, w, b):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_elements = x.shape
    eps = 1e-5
    _layernorm[(n_rows,)](x, w, b, out, n_elements, eps, BLOCK=triton.next_power_of_2(n_elements))
    return out
