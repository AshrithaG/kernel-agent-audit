import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def _bias_gelu_kernel(x_ptr, b_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    
    # Load x and b, ensuring they are in float32
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Add bias to x
    x_plus_b = x + b
    
    # GELU approximation using tanh
    cdf = 0.5 * (1.0 + libdevice.tanh((x_plus_b / 1.4142135623730951) * (1.0 + 0.044715 * x_plus_b * x_plus_b)))
    gelu_approx = x_plus_b * cdf
    
    # Store the result back to out_ptr
    tl.store(out_ptr + row * n_cols + cols, gelu_approx.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, b):
    x = x.contiguous()
    b = b.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    _bias_gelu_kernel[(n_rows,)](x, b, out, n_cols, BLOCK=triton.next_power_of_2(n_cols))
    return out
