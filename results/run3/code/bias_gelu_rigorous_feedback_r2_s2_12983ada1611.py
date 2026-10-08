import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh

# Define the block size
BLOCK_SIZE = 128

@triton.jit
def _bias_gelu_kernel(x_ptr, b_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    # Shared memory for caching b
    b_shared = tl.zeros((BLOCK,), dtype=tl.float32)
    
    # Load b into shared memory
    b_index = tl.arange(0, BLOCK)
    b_mask = b_index < n_cols
    tl.store(b_shared, tl.load(b_ptr + b_index, mask=b_mask, other=0.0).to(tl.float32), mask=b_mask)
    tl.sync_threads()
    
    # Calculate the row index
    row = tl.program_id(0)
    
    # Calculate the column indices
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    
    # Load x
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Add bias
    b = tl.load(b_shared + cols, mask=mask, other=0.0).to(tl.float32)
    x_plus_b = x + b
    
    # GELU approximation using a more accurate polynomial
    cdf = 0.5 * (1.0 + tanh(0.7978845608028654 * (x_plus_b + 0.044715 * x_plus_b * x_plus_b * x_plus_b)))
    gelu_approx = x_plus_b * cdf
    
    # Store the result
    tl.store(out_ptr + row * n_cols + cols, gelu_approx.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, b):
    x = x.contiguous()
    b = b.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    _bias_gelu_kernel[(n_rows,)](x, b, out, n_cols, BLOCK=BLOCK_SIZE)
    return out
