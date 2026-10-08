import torch
import triton
import triton.language as tl

@triton.jit
def _row_mean(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Use shared memory to cache the data
    shared_mem = tl.zeros(BLOCK, dtype=tl.float32)
    shared_mem[cols] = x
    
    # Reduce the sum using shared memory
    sum_val = tl.sum(shared_mem, axis=0)
    
    # Store the result
    tl.store(out_ptr + row, (sum_val / n_cols).to(out_ptr.dtype.element_ty), mask=True)

def kernel(x):
    x = x.contiguous()
    out = torch.empty((x.shape[0],), dtype=torch.float16, device=x.device)
    n_rows, n_cols = x.shape
    BLOCK = 1024  # Larger block size
    grid = (n_rows,)
    _row_mean[grid](x, out, n_cols, BLOCK=BLOCK)
    return out
