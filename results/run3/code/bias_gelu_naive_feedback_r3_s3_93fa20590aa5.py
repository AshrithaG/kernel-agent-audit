import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh

# Define a vectorized type for float32
float32x4 = tl.float32 * 4

@triton.jit
def _bias_gelu_kernel(x_ptr, b_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    # Load data into vectorized registers
    x_vec = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(float32x4)
    b_vec = tl.load(b_ptr + cols, mask=mask, other=0.0).to(float32x4)
    
    # Constants for GELU approximation
    k = 0.7978845608028654
    k_vec = tl.full([BLOCK], k, dtype=float32x4)
    
    # Compute x + b
    x_plus_b_vec = x_vec + b_vec
    
    # Compute y = k * (x + b)
    y_vec = k_vec * x_plus_b_vec
    
    # Compute y^2
    y2_vec = y_vec * y_vec
    
    # Compute the GELU approximation
    inner_tanh = y_vec * (1.0 + 0.044715 * y2_vec)
    gelu_approx_vec = 0.5 * x_plus_b_vec * (1.0 + tanh(inner_tanh))
    
    # Store the result back to memory
    tl.store(out_ptr + row * n_cols + cols, gelu_approx_vec.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, b):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    # Use a larger block size for better parallelism
    BLOCK_SIZE = triton.next_power_of_2(max(n_cols, 128))
    _bias_gelu_kernel[(n_rows,)](x, b, out, n_cols, BLOCK=BLOCK_SIZE)
    return out
