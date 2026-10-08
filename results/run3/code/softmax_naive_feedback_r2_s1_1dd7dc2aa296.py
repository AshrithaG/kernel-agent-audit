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
    
    # Compute exp and sum of exps
    exp_x = tl.exp(x)
    
    # Reduce sum of exps using a loop
    sum_exp_x = tl.zeros([BLOCK], dtype=tl.float32)
    stride = BLOCK >> 1
    while stride > 0:
        if stride <= BLOCK:
            sum_exp_x += tl.shift_right(exp_x, stride)
        stride >>= 1
    
    # Compute softmax
    softmax_x = exp_x / sum_exp_x
    
    tl.store(out_ptr + row * n_cols + cols, softmax_x.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    _softmax[(n_rows,)](x, out, n_cols, BLOCK=triton.next_power_of_2(n_cols))
    return out
