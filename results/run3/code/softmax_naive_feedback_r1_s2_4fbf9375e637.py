import torch
import triton
import triton.language as tl

@triton.jit
def _softmax(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    # Define the width of the vectorized operations
    VecSize = 8
    Vec = lambda x: x.to(tl.float32x8)
    
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK, step=VecSize)
    mask = col_offsets < n_cols
    
    # Load input data
    x_vec = tl.load(x_ptr + row * n_cols + col_offsets, mask=mask, other=float('-inf'))
    
    # Compute max for numerical stability
    max_x_vec = tl.max(x_vec, axis=0)
    x_vec = x_vec - max_x_vec
    
    # Compute exp and sum of exps
    exp_x_vec = tl.exp(x_vec)
    sum_exp_x_vec = tl.sum(exp_x_vec, axis=0)
    
    # Compute softmax
    softmax_x_vec = exp_x_vec / sum_exp_x_vec
    
    # Store the results back into the output tensor
    tl.store(out_ptr + row * n_cols + col_offsets, softmax_x_vec, mask=mask)

def kernel(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    # Ensure BLOCK is a multiple of VecSize
    BLOCK = BLOCK + (VecSize - BLOCK % VecSize) % VecSize
    _softmax[(n_rows,)](x, out, n_cols, BLOCK=BLOCK)
    return out
