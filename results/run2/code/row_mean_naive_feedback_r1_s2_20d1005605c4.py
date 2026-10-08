import torch
import triton
import triton.language as tl

@triton.jit
def mean_kernel(x_ptr, out_ptr, N, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # Calculate the row index
    row_idx = pid // BLOCK_SIZE
    # Calculate the block index within the row
    block_idx = pid % BLOCK_SIZE
    # Calculate the start index of the block within the row
    block_start = block_idx * BLOCK_SIZE
    # Calculate the offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to ensure we don't read out of bounds
    mask = offsets < M
    # Load the elements from the current block into shared memory
    x = tl.load(x_ptr + row_idx * M + offsets, mask=mask, other=0.0)
    # Compute the sum and count of the elements in the block
    sum_x = tl.sum(x, axis=0)
    count = tl.sum(mask, axis=0)
    # Reduce the sums and counts across the blocks using warp-level operations
    sum_x = tl.sum(sum_x, axis=0)
    count = tl.sum(count, axis=0)
    # Compute the mean
    mean = sum_x / count
    # Store the mean in the output tensor
    if block_idx == 0:
        tl.store(out_ptr + row_idx, mean)

def kernel(x):
    N, M = x.shape
    BLOCK_SIZE = 128
    out = torch.empty(N, dtype=x.dtype, device=x.device)
    grid = lambda opt: (N * triton.cdiv(M, BLOCK_SIZE),)
    mean_kernel[grid](x, out, N, M, BLOCK_SIZE)
    return out

# Example usage:
# x = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
# result = kernel(x)
# print(result)
