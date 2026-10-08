import torch
import triton
import triton.language as tl
import math

# Reference implementation for verification
def reference(x):
    n = x.shape[-1] // 2
    return (torch.nn.functional.silu(x[..., :n].float()) * x[..., n:].float()).to(x.dtype)

# Triton kernel for SwiGLU
@triton.jit
def swiglu_kernel(
    X, Y,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    **META
):
    pid = tl.program_id(axis=0)
    stride = tl.num_programs_in_dim(axis=0)
    off_x = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off_x < N
    x = tl.load(X + off_x, mask=mask, other=0.0)
    x1 = x[:N//2]
    x2 = x[N//2:]
    x1 = x1 * (1 / (1 + tl.exp(-x1)))
    y = x1 * x2
    tl.store(Y + off_x, y, mask=mask)

# Triton wrapper function
def kernel(x):
    N = x.shape[-1]
    BLOCK_SIZE = 128
    grid = lambda META: (triton.cdiv(N, BLOCK_SIZE),)
    y = torch.empty_like(x)
    swiglu_kernel[grid](x, y, N=N, BLOCK_SIZE=BLOCK_SIZE)
    return y

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
    ref_output = reference(x)
    triton_output = kernel(x)
    print("Reference output:", ref_output)
    print("Triton output:", triton_output)
    assert torch.allclose(ref_output, triton_output, atol=1e-3, rtol=1e-3)
