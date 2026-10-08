import torch
import triton
import triton.language as tl

# Define the block size
BLOCK_SIZE = 128

@triton.jit
def _row_mean_optimized(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    # Each thread block processes one row
    row = tl.program_id(0)
    # Each thread processes multiple elements per iteration
    pid = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    # Calculate the starting column index for this thread
    start_col = pid * BLOCK
    # Load data into shared memory
    sdata = tl.zeros((BLOCK,), dtype=tl.float32)
    # Mask to handle cases where the number of columns is not a multiple of BLOCK
    mask = cols < n_cols - start_col
    # Load data from global memory to shared memory
    sdata = tl.load(x_ptr + row * n_cols + start_col + cols, mask=mask, other=0.0).to(tl.float32)
    # Reduce the sum of the elements in the block
    sum_val = tl.sum(sdata, axis=0)
    # Store the result in global memory
    if pid == 0:
        tl.store(out_ptr + row, (sum_val / n_cols).to(out_ptr.dtype.element_ty), mask=True)

def kernel(x):
    x = x.contiguous()
    out = torch.empty((x.shape[0],), dtype=torch.float16, device=x.device)
    n_rows, n_cols = x.shape
    # Calculate the number of threads per block
    num_threads_per_block = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (n_rows, num_threads_per_block)
    _row_mean_optimized[grid](x, out, n_cols, BLOCK=BLOCK_SIZE)
    return out
