import torch
import triton
import triton.language as tl

# Define the block size
BLOCK = 128

@triton.jit
def _softmax(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    # Each row is processed by a separate program instance
    row = tl.program_id(0)
    # Vectorized load: process 4 elements at a time
    cols = tl.arange(0, BLOCK, 4)
    mask = cols < n_cols
    # Load 4 elements at a time
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=float('-inf')).to(tl.float32)
    
    # Compute max for numerical stability
    max_x = tl.max(x, axis=0)
    x = x - max_x
    
    # Compute exp and sum of exps
    exp_x = tl.exp(x)
    sum_exp_x = tl.sum(exp_x, axis=0)
    
    # Compute softmax
    softmax_x = exp_x / sum_exp_x
    
    # Store 4 elements at a time
    tl.store(out_ptr + row * n_cols + cols, softmax_x.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    # Launch the kernel with a grid size equal to the number of rows
    # and a block size of BLOCK
    _softmax[(n_rows,)](x, out, n_cols, BLOCK=BLOCK)
    return out
