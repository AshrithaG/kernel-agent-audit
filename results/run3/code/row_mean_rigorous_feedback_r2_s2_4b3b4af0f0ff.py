import torch
import triton
import triton.language as tl

@triton.jit
def _row_mean(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Vectorized reduction
    partial_sums = tl.zeros(BLOCK, dtype=tl.float32)
    for i in range(0, BLOCK, 32):
        partial_sums[i:i+32] += x[i:i+32]
    
    # Further reduce the partial sums
    for stride in [16, 8, 4, 2]:
        if BLOCK >= stride * 2:
            partial_sums[:stride] += tl.shift_right(partial_sums, stride)[:stride]
    
    mean = partial_sums[0] / n_cols
    tl.store(out_ptr + row, mean.to(out_ptr.dtype.element_ty), mask=True)

def kernel(x):
    x = x.contiguous()
    out = torch.empty((x.shape[0],), dtype=torch.float16, device=x.device)
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    _row_mean[grid](x, out, n_cols, BLOCK=BLOCK)
    return out
