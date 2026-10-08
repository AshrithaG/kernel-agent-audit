import torch
import triton
import triton.language as tl

@triton.jit
def _softmax(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK, dtype=tl.int32)
    mask = cols < n_cols
    
    # Load input data
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=float('-inf')).to(tl.float32)
    
    # Compute max for numerical stability
    max_x = tl.max(x, axis=0)
    x = x - max_x
    
    # Compute exp and sum of exps
    exp_x = tl.exp(x)
    sum_exp_x = tl.sum(exp_x, axis=0)
    
    # Compute softmax
    softmax_x = exp_x / sum_exp_x
    
    # Store results
    tl.store(out_ptr + row * n_cols + cols, softmax_x.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = BLOCK_SIZE // 32
    grid = (n_rows,)
    _softmax[grid](x, out, n_cols, BLOCK=BLOCK_SIZE, num_warps=num_warps)
    return out
