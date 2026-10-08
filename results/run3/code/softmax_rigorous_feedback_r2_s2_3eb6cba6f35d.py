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
    
    # Compute sum of exps using shared memory for reduction
    smem = tl.zeros(BLOCK, dtype=tl.float32)
    smem[cols] = exp_x
    tl.block_sync()
    
    for i in range(0, BLOCK, 32):
        smem[cols] += smem[cols + i]
    tl.block_sync()
    
    for i in range(0, BLOCK, 16):
        smem[cols] += smem[cols + i]
    tl.block_sync()
    
    for i in range(0, BLOCK, 8):
        smem[cols] += smem[cols + i]
    tl.block_sync()
    
    for i in range(0, BLOCK, 4):
        smem[cols] += smem[cols + i]
    tl.block_sync()
    
    for i in range(0, BLOCK, 2):
        smem[cols] += smem[cols + i]
    tl.block_sync()
    
    sum_exp_x = smem[0]
    
    # Compute softmax
    softmax_x = exp_x / sum_exp_x
    
    tl.store(out_ptr + row * n_cols + cols, softmax_x.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    _softmax_kernel[(n_rows,)](x, out, n_cols, BLOCK=BLOCK_SIZE)
    return out
