import torch
import triton
import triton.language as tl

@triton.jit
def _row_mean(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    # Define the row and column indices
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    
    # Load data into shared memory
    x_shared = tl.zeros((BLOCK,), dtype=tl.float32)
    tl.store(x_shared, tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32), mask=mask)
    
    # Perform reduction in shared memory
    sum_val = tl.sum(x_shared, axis=0)
    
    # Store the result back to global memory
    tl.store(out_ptr + row, sum_val / n_cols, mask=True)

def kernel(x):
    x = x.contiguous()
    out = torch.empty((x.shape[0],), dtype=torch.float16, device=x.device)
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    _row_mean[grid](x, out, n_cols, BLOCK=BLOCK)
    return out
