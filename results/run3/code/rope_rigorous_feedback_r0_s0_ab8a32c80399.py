import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def _rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, tokens, heads, dim, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = dim // 2
    cols = tl.arange(0, BLOCK)
    mask = cols < d
    
    # Load x1 and x2
    x1_ptr = x_ptr + token * heads * dim + head * dim + cols
    x2_ptr = x_ptr + token * heads * dim + head * dim + d + cols
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr, mask=mask, other=0.0).to(tl.float32)
    
    # Load cos and sin
    c_ptr = cos_ptr + token * d + cols
    s_ptr = sin_ptr + token * d + cols
    c = tl.load(c_ptr, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_ptr, mask=mask, other=0.0).to(tl.float32)
    
    # Compute the rotated values
    x1_rotated = x1 * c - x2 * s
    x2_rotated = x2 * c + x1 * s
    
    # Store the results
    tl.store(out_ptr + token * heads * dim + head * dim + cols, x1_rotated.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(out_ptr + token * heads * dim + head * dim + d + cols, x2_rotated.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, cos, sin):
    x = x.contiguous()
    out = torch.empty_like(x)
    tokens, heads, dim = x.shape
    BLOCK = triton.next_power_of_2(dim // 2)
    _rope_kernel[(tokens, heads), (BLOCK,)](x, cos, sin, out, tokens, heads, dim, BLOCK=BLOCK)
    return out
