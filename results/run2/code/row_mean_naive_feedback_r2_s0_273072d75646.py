import torch
import triton
import triton.language as tl

@triton.jit
def mean_kernel(x_ptr, out_ptr, stride_x, N, M):
    # Each thread computes the sum of a portion of the row
    row_id = tl.program_id(axis=0)
    row_sum = tl.zeros(1, dtype=tl.float32)
    
    # Each thread processes a segment of the row
    for j in range(M // tl.num_warps()):
        idx = row_id * M + j * tl.num_warps()
        if idx < N * M:
            row_sum += tl.load(x_ptr + idx)
    
    # Sum the partial sums across warps in the block
    row_sum = tl.sum(row_sum, axis=0)
    
    # Compute the mean and store it
    row_mean = row_sum / M
    tl.store(out_ptr + row_id, row_mean)

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
