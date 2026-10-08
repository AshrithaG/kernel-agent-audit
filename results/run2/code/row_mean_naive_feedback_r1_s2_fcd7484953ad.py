import torch
import triton
import triton.language as tl

@triton.jit
def mean_kernel(x_ptr, out_ptr, stride_x, N, M):
    # Each thread computes the sum for one row
    row_idx = tl.program_id(axis=0)
    if row_idx >= N:
        return
    row_sum = tl.zeros([M], dtype=tl.float32)
    for j in range(M):
        row_sum[j] += tl.load(x_ptr + row_idx * stride_x + j)
    row_mean = row_sum / M
    tl.store(out_ptr + row_idx, row_mean)

BLOCK_M = 128

def kernel(x):
    N, M = x.shape
    out = torch.empty(N, dtype=torch.float32, device=x.device)
    grid = lambda META: (triton.cdiv(N, BLOCK_M),)
    mean_kernel[grid](x, out, x.stride(0), N, M)
    return out.to(x.dtype)

# Example usage:
# x = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
# result = kernel(x)
# print(result)
