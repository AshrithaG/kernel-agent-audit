import torch
import triton
import triton.language as tl

@triton.jit
def swiglu_kernel(x_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    block_start = pid * BLOCK_SIZE + bid * M * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    first_half = x[:M]
    second_half = x[M:]
    first_half = first_half * (1 / (1 + tl.exp(-first_half)))
    y = first_half * second_half
    tl.store(y_ptr + offsets, y, mask=mask)

def kernel(x):
    B, N, H = x.shape
    y = torch.empty_like(x)
    grid = lambda opt: (
        triton.cdiv(N, opt.BLOCK_SIZE),
        B * H,
    )
    swiglu_kernel[grid](x, y, H, N, BLOCK_SIZE=128)
    return y

# Example usage:
if __name__ == "__main__":
    x = torch.randn(1024, 2048, device='cuda', dtype=torch.float16)
    y = kernel(x)
    print(y)
