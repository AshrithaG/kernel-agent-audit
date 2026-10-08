import torch
import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    
    # Load input values
    input_values = tl.load(input_ptr + row * n_cols + cols, mask=mask, other=float('-inf')).to(tl.float32)
    
    # Compute max value for numerical stability
    max_value = tl.max(input_values, axis=0)
    
    # Subtract max value and compute exp
    exp_values = tl.exp(input_values - max_value)
    
    # Compute sum of exp values
    sum_exp_values = tl.sum(exp_values, axis=0)
    
    # Compute softmax
    softmax_values = exp_values / sum_exp_values
    
    # Store results
    tl.store(output_ptr + row * n_cols + cols, softmax_values.to(output_ptr.dtype.element_ty), mask=mask)

def kernel(input):
    input = input.contiguous()
    output = torch.empty_like(input)
    n_rows, n_cols = input.shape
    _softmax_kernel[(n_rows,)](input, output, n_cols, BLOCK=triton.next_power_of_2(n_cols))
    return output
