import torch
import triton
import triton.language as tl

# Define the block size
BLOCK_SIZE = 128

@triton.jit
def _softmax(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    
    # Load data into shared memory
    x_shared = tl.full((BLOCK,), float('-inf'), dtype=tl.float32)
    tl.store(x_shared + cols, tl.load(x_ptr + row * n_cols + cols, mask=mask, other=float('-inf')), mask=mask)
    
    # Compute max for numerical stability
    max_x = tl.max(x_shared, axis=0)
    x_shared = x_shared - max_x
    
    # Compute exp and sum of exps using shared memory
    exp_x_shared = tl.exp(x_shared)
    sum_exp_x = tl.sum(exp_x_shared, axis=0)
    
    # Compute softmax
    softmax_x_shared = exp_x_shared / sum_exp_x
    
    # Store the results back into the output tensor
    tl.store(out_ptr + row * n_cols + cols, softmax_x_shared.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    _softmax[(n_rows,)](x, out, n_cols, BLOCK=BLOCK_SIZE)
    return out
