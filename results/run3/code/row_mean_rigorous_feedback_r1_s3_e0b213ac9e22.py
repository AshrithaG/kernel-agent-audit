import torch
import triton
import triton.language as tl

@triton.jit
def _row_mean_optimized(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    # Vectorized load: load multiple elements per thread
    vec_size = 4
    vec_cols = tl.arange(0, BLOCK // vec_size)
    mask = vec_cols < (n_cols // vec_size)
    x_vec = tl.load(x_ptr + row * n_cols + vec_cols * vec_size, mask=mask, other=0.0).to(tl.float32)
    
    # Sum the elements within each vector
    sum_vec = tl.sum(x_vec, axis=0)
    
    # Reduce the sums across the block
    sum_block = tl.sum(sum_vec, axis=0)
    
    # Compute the mean
    mean = sum_block / n_cols
    
    # Store the result
    tl.store(out_ptr + row, mean.to(out_ptr.dtype.element_ty), mask=True)

def kernel(x):
    x = x.contiguous()
    out = torch.empty((x.shape[0],), dtype=torch.float16, device=x.device)
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    if BLOCK > 256:
        BLOCK = 256  # Limit block size to 256 to avoid excessive register usage
    _row_mean_optimized[(n_rows,)](x, out, n_cols, BLOCK=BLOCK)
    return out
