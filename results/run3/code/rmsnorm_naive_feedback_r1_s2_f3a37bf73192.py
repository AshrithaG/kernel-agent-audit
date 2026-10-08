import torch
import triton
import triton.language as tl

# Define the block size
BLOCK_SIZE = 128

@triton.jit
def _rmsnorm(x_ptr, w_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    # Each row is processed by a separate program instance
    row = tl.program_id(0)
    # Each block processes a subset of columns
    cols = tl.arange(0, BLOCK)
    # Mask to ensure we don't access out of bounds
    mask = cols < n_cols
    
    # Load x and w into registers
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute the mean of squares using shared memory for reduction
    # Allocate shared memory
    smem = tl.zeros([BLOCK], dtype=tl.float32)
    # Compute local sum of squares
    local_sum_squares = tl.sum(x * x, axis=0)
    # Write to shared memory
    tl.store(smem, local_sum_squares)
    # Synchronize threads to ensure all writes are complete
    tl.barrier()
    
    # Reduce the sum of squares in shared memory
    for i in range(BLOCK // 2):
        if cols < BLOCK // 2:
            smem[i] += smem[i + BLOCK // 2]
    # Synchronize again after reduction
    tl.barrier()
    
    # Only the first thread in the block will have the final sum of squares
    if cols == 0:
        mean_squares = smem[0] / n_cols
    else:
        mean_squares = tl.load(smem, mask=cols == 0, other=0.0)
    
    # Broadcast mean_squares to all threads
    mean_squares = tl.broadcast_to(mean_squares, [BLOCK])
    
    # Compute rsqrt
    rsqrt_mean_squares = tl.rsqrt(mean_squares + 1e-6)
    
    # Normalize and scale
    normalized_x = x * rsqrt_mean_squares
    scaled_x = normalized_x * w
    
    # Store the result
    tl.store(out_ptr + row * n_cols + cols, scaled_x.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, w):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    # Launch the kernel with the appropriate grid and block size
    _rmsnorm[(n_rows,)](x, w, out, n_cols, BLOCK=BLOCK_SIZE)
    return out
