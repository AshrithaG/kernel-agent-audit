import torch
import triton
import triton.language as tl

@triton.jit
def mean_kernel(x_ptr, out_ptr, stride_x, N, M):
    pid = tl.program_id(axis=0)
    rows_per_block = BLOCK_M
    row_start = pid * rows_per_block
    row_end = min(row_start + rows_per_block, N)
    for i in range(row_start, row_end):
        row_sum = tl.zeros([M], dtype=tl.float32)
        for j in range(M):
            row_sum[j] += tl.load(x_ptr + i * stride_x + j)
        row_mean = row_sum / M
        tl.store(out_ptr + i, row_mean)

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
