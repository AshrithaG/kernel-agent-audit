import torch
import triton
import triton.language as tl

@triton.jit
def _softmax(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=float('-inf')).to(tl.float32)
    
    # Compute max for numerical stability
    max_x = tl.max(x, axis=0)
    x = x - max_x
    
    # Compute exp and sum of exps with loop unrolling
    exp_x = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, BLOCK, 4):
        exp_x[i:i+4] = tl.exp(x[i:i+4])
    sum_exp_x = tl.sum(exp_x, axis=0)
    
    # Compute softmax
    softmax_x = exp_x / sum_exp_x
    
    tl.store(out_ptr + row * n_cols + cols, softmax_x.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    if BLOCK > 128:
        BLOCK = 128  # Limit block size to 128 to avoid excessive register usage
    _softmax[(n_rows,)](x, out, n_cols, BLOCK=BLOCK)
    return out
