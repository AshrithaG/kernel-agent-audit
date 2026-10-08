import torch
import triton
import triton.language as tl
from triton.language import extra

@triton.jit
def _add_rmsnorm(x_ptr, r_ptr, w_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    elements = tl.arange(0, BLOCK)
    mask = elements < n_elements
    x = tl.load(x_ptr + pid * n_elements + elements, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + pid * n_elements + elements, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr, mask=mask, other=0.0).to(tl.float32)
    
    h = x + r
    squared_sum = tl.sum(h * h, axis=-1)
    rms = tl.rsqrt(squared_sum + 1e-6)
    normalized_h = h * rms
    result = normalized_h * w
    
    tl.store(out_ptr + pid * n_elements + elements, result.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, r, w):
    x = x.contiguous()
    r = r.contiguous()
    w = w.contiguous()
    out = torch.empty_like(x)
    n_elements = x.size(-1)
    grid = (x.size(0),)
    _add_rmsnorm[grid](x, r, w, out, n_elements, BLOCK=triton.next_power_of_2(n_elements))
    return out
