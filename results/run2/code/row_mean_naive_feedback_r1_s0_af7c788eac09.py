import torch
import triton
import triton.language as tl

@triton.jit
def mean_kernel(x_ptr, out_ptr, N, M, BLOCK_SIZE: tl.constexpr):
    # Each thread processes one row
    row_idx = tl.program_id(axis=0)
    if row_idx >= N:
        return
    
    # Load the entire row into shared memory
    row_offset = row_idx * M
    row_ptr = x_ptr + row_offset
    
    # Initialize sum and count
    sum_x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    
    # Process elements in blocks
    for i in range(0, M, BLOCK_SIZE):
        # Load elements from the row
        offsets = tl.arange(0, BLOCK_SIZE)
        x = tl.load(row_ptr + offsets, mask=offsets < M - i, other=0.0)
        
        # Update sum and count
        sum_x += x
        count += tl.where(offsets < M - i, 1, 0)
    
    # Reduce sum and count across the block
    sum_x = tl.sum(sum_x, axis=0)
    count = tl.sum(count, axis=0)
    
    # Compute mean
    mean = sum_x / count
    
    # Store the mean in the output tensor
    tl.store(out_ptr + row_idx, mean)

def kernel(x):
    N, M = x.shape
    BLOCK_SIZE = 128
    out = torch.empty(N, dtype=x.dtype, device=x.device)
    grid = lambda opt: (N,)
    mean_kernel[grid](x, out, N, M, BLOCK_SIZE)
    return out

# Example usage:
# x = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
# result = kernel(x)
# print(result)
