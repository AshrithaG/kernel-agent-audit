import torch
import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=float('-inf')).to(tl.float32)
    
    # Compute max for numerical stability
    max_x = tl.max(x, axis=0)
    x = x - max_x
    
    # Compute exp
    exp_x = tl.exp(x)
    
    # Compute sum of exps using a more efficient reduction
    # First, reduce within the block
    partial_sum = tl.sum(exp_x, axis=0)
    
    # Reduce across blocks (if needed)
    # This part is simplified for now, assuming one block per row
    # For larger blocks, you would need to add a cross-block reduction step
    
    # Compute softmax
    softmax_x = exp_x / partial_sum
    
    tl.store(out_ptr + row * n_cols + cols, softmax_x.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    _softmax_kernel[(n_rows,)](x, out, n_cols, BLOCK=BLOCK_SIZE)
    return out
