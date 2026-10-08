import torch
import triton
import triton.language as tl

@triton.jit
def mean_kernel(x_ptr, out_ptr, stride_x, N, M):
    # Each thread computes the mean of one row
    row_idx = tl.program_id(axis=0)
    
    # Load the row into shared memory
    row = tl.zeros([M], dtype=tl.float32)
    for j in range(M):
        row[j] += tl.load(x_ptr + row_idx * stride_x + j)
    
    # Compute the mean of the row
    row_mean = row / M
    
    # Store the result
    tl.store(out_ptr + row_idx, row_mean)

BLOCK_M = 128

def kernel(x):
    N, M = x.shape
    out = torch.empty(N, dtype=torch.float16, device=x.device)
    grid = lambda META: (triton.cdiv(N, BLOCK_M),)
    mean_kernel[grid](x, out, x.stride(0), N, M)
    return out

# Example usage:
# x = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
# result = kernel(x)
# print(result)
