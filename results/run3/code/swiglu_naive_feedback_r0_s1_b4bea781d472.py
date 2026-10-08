import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import sigmoid

@triton.jit
def _swiglu_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    n = n_elements // 2
    
    # Load the first half and apply SiLU
    first_half_idx = idx[:n]
    first_half = tl.load(x_ptr + first_half_idx, mask=mask[:n], other=0.0).to(tl.float32)
    sielu = first_half * sigmoid(first_half)
    
    # Load the second half
    second_half_idx = idx[n:]
    second_half = tl.load(x_ptr + second_half_idx, mask=mask[n:], other=0.0).to(tl.float32)
    
    # Multiply SiLU of first half with second half
    result = sielu * second_half
    
    # Store the result
    tl.store(out_ptr + idx, result.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK = triton.next_power_of_2(n_elements)
    _swiglu_kernel[(n_elements // BLOCK,)](x, out, n_elements, BLOCK=BLOCK)
    return out
