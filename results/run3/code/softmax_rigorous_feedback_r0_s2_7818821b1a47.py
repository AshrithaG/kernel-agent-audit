import torch
import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(input_ptr, output_ptr, max_val_ptr, sum_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    # Load input values
    input_vals = tl.load(input_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)

    # Subtract max value to avoid overflow
    max_val = tl.load(max_val_ptr + row, other=0.0).to(tl.float32)
    exp_vals = tl.exp(input_vals - max_val)

    # Store exp values back to output for reduction
    tl.store(output_ptr + row * n_cols + cols, exp_vals, mask=mask)

    # Reduce to get the sum of exp values
    sum_val = tl.sum(exp_vals, axis=0)
    tl.atomic_add(sum_ptr + row, sum_val)

@triton.jit
def _normalize_kernel(output_ptr, sum_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    # Load exp values and sum
    exp_vals = tl.load(output_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    sum_val = tl.load(sum_ptr + row, other=0.0).to(tl.float32)

    # Normalize by sum
    softmax_vals = exp_vals / sum_val

    # Store the final softmax values
    tl.store(output_ptr + row * n_cols + cols, softmax_vals, mask=mask)

def kernel(x):
    x = x.contiguous()
    n_rows, n_cols = x.shape

    # Allocate buffers
    max_vals = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    sum_vals = torch.zeros((n_rows,), device=x.device, dtype=torch.float32)
    exp_vals = torch.empty_like(x, dtype=torch.float32)

    # Compute max values for each row
    _softmax_kernel[(n_rows,)](x, exp_vals, max_vals, sum_vals, n_cols, BLOCK=triton.next_power_of_2(n_cols))

    # Normalize exp values
    _normalize_kernel[exp_vals, sum_vals, n_cols, BLOCK=triton.next_power_of_2(n_cols)]

    return exp_vals.to(x.dtype)
