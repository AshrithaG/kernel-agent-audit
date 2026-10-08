import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh

@triton.jit
def _bias_gelu_kernel(x_ptr, b_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    # Constants
    k = 0.7978845608028654
    one_over_sqrt_two = 1.0 / tl.sqrt(tl.float32(2.0))
    
    # Program ID and block indices
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    
    # Load data
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute x + b
    x_plus_b = x + b
    
    # Compute the GELU approximation
    y = x_plus_b * k
    y2 = y * y
    gelu_approx = 0.5 * x_plus_b * (1.0 + tanh(y * (1.0 + 0.044715 * y2)))
    
    # Store the result
    tl.store(out_ptr + row * n_cols + cols, gelu_approx.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, b):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 128  # Adjust block size for optimal performance
    _bias_gelu_kernel[(n_rows,)](x, b, out, n_cols, BLOCK=BLOCK_SIZE)
    return out
