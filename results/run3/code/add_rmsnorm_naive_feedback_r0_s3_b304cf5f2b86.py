import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def _add_rmsnorm_kernel(x_ptr, r_ptr, w_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # Calculate the row and column indices for the current program instance
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_elements
    
    # Load the data from the input tensors
    x = tl.load(x_ptr + row * n_elements + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * n_elements + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Add the residual
    h = x + r
    
    # Compute the square root of the mean of squares plus epsilon
    mean_squares = tl.sum(h * h, axis=1, keepdims=True) / n_elements
    rms = tl.sqrt(mean_squares + 1e-6)
    
    # Load the weight tensor
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Apply the normalization and scaling
    normalized_h = h / rms
    scaled_h = normalized_h * w
    
    # Store the result back to the output tensor
    tl.store(out_ptr + row * n_elements + cols, scaled_h.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, r, w):
    x = x.contiguous()
    r = r.contiguous()
    w = w.contiguous()
    out = torch.empty_like(x)
    n_rows, n_elements = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    _add_rmsnorm_kernel[(n_rows,)](x, r, w, out, n_elements, BLOCK=BLOCK_SIZE)
    return out
