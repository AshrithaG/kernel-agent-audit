import torch
import triton
import triton.language as tl

@triton.jit
def mean_kernel(x_ptr, out_ptr, N, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sum_x = tl.sum(x, axis=0)
    count = tl.sum(mask, axis=0)
    mean = sum_x / count
    tl.store(out_ptr + pid, mean)

def kernel(x):
    N, M = x.shape
    BLOCK_SIZE = 128
    out = torch.empty(N, dtype=x.dtype, device=x.device)
    grid = lambda opt: (triton.cdiv(N, BLOCK_SIZE),)
    mean_kernel[grid](x, out, N, M, BLOCK_SIZE)
    return out

# Example usage:
# x = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
# result = kernel(x)
# print(result)
