import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh

@triton.jit
def _swiglu(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    
    n_half = n_elements // 2
    first_half = idx < n_half
    second_half = idx >= n_half
    
    # Compute SiLU for the first half
    siLu = x * tanh(tl.log1p(x))
    
    # Multiply the first half by the second half
    result = tl.where(first_half, siLu, x) * tl.where(second_half, x, 1.0)
    
    tl.store(out_ptr + idx, result.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK = triton.next_power_of_2(n_elements)
    _swiglu[(n_elements // BLOCK,)](x, out, n_elements, BLOCK=BLOCK)
    return out
