import torch
import triton
import triton.language as tl

@triton.jit
def _row_mean(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.sum(x, axis=0)
    mean_x = sum_x / n_cols
    tl.store(out_ptr + row, mean_x.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x):
    x = x.contiguous()
    out = torch.empty((x.shape[0],), dtype=torch.float16, device=x.device)
    n_rows, n_cols = x.shape
    _row_mean[(n_rows,)](x, out, n_cols, BLOCK=triton.next_power_of_2(n_cols))
    return out
