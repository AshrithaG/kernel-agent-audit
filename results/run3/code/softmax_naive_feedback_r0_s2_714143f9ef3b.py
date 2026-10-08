import torch
import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(input_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute max for numerical stability
    max_x = tl.max(x, axis=0)
    x_minus_max = x - max_x
    
    # Compute exp and sum of exps
    exp_x = tl.exp(x_minus_max)
    sum_exp_x = tl.sum(exp_x, axis=0)
    
    # Compute softmax
    softmax_values = exp_x / sum_exp_x
    
    # Store the results
    tl.store(output_ptr + row * n_cols + cols, softmax_values.to(output_ptr.dtype.element_ty), mask=mask)

def kernel(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    _softmax_kernel[(n_rows,)](x, out, n_cols, BLOCK=triton.next_power_of_2(n_cols))
    return out
