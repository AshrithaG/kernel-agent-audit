import torch
import triton
import triton.language as tl

@triton.jit
def _row_mean_optimized(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK, dtype=tl.int32)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Vectorized reduction
    sum_acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, BLOCK, 32):
        sum_acc += tl.sum(x[i:i+32], axis=0)
    
    # Final reduction
    sum_acc = tl.sum(sum_acc, axis=0) / n_cols
    tl.store(out_ptr + row, sum_acc.to(out_ptr.dtype.element_ty), mask=True)

def kernel(x):
    x = x.contiguous()
    out = torch.empty((x.shape[0],), dtype=torch.float16, device=x.device)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024
    _row_mean_optimized[(n_rows,)](x, out, n_cols, BLOCK=BLOCK_SIZE)
    return out
