import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh

# Define a more accurate polynomial for GELU approximation
@triton.jit
def gelu_approx(x):
    k = 0.7978845608028654
    return 0.5 * x * (1.0 + tanh(k * (x + 0.044715 * x * x * x)))

@triton.jit
def _bias_gelu_kernel(x_ptr, b_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    
    # Load x and b
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Add bias
    x_plus_b = x + b
    
    # Apply GELU approximation
    gelu_result = gelu_approx(x_plus_b)
    
    # Store the result
    tl.store(out_ptr + row * n_cols + cols, gelu_result.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, b):
    x = x.contiguous()
    b = b.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    _bias_gelu_kernel[(n_rows,)](x, b, out, n_cols, BLOCK=triton.next_power_of_2(n_cols))
    return out
