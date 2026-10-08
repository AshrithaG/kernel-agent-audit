import torch
import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=float('-inf')).to(tl.float32)
    
    # Find the max value in the row to improve numerical stability
    max_val = tl.max(x, axis=0)
    x = x - max_val
    
    # Compute the exponential of the shifted values
    exp_x = tl.exp(x)
    
    # Sum the exponentials
    sum_exp_x = tl.sum(exp_x, axis=0)
    
    # Compute the softmax values
    softmax_x = exp_x / sum_exp_x
    
    # Store the results
    tl.store(out_ptr + row * n_cols + cols, softmax_x.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    _softmax_kernel[(n_rows,)](x, out, n_cols, BLOCK=triton.next_power_of_2(n_cols))
    return out
