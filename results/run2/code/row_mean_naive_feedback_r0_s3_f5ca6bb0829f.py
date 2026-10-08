import torch
import triton
import triton.language as tl
import math

@triton.jit
def mean_kernel(
    X,  # pointer to input matrix
    Y,  # pointer to output vector
    N,  # number of rows
    M,  # number of columns
    BLOCK_SIZE_M: tl.constexpr,  # number of rows processed by each threadblock
    BLOCK_SIZE_N: tl.constexpr,  # number of columns processed by each threadblock
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    row = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask = (row < N) & (col < M)
    x = tl.load(X + row[:, None] * M + col[None, :], mask=mask, other=0.0)
    sum_val = tl.sum(x, axis=1)
    count = tl.sum(mask, axis=1)
    mean = sum_val / count
    tl.store(Y + row, mean)

def kernel(X):
    N, M = X.shape
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 128
    grid = (
        triton.cdiv(N, BLOCK_SIZE_M),
        triton.cdiv(M, BLOCK_SIZE_N),
    )
    Y = torch.empty((N,), device=X.device, dtype=torch.float32)
    mean_kernel[grid](X, Y, N, M, BLOCK_SIZE_M, BLOCK_SIZE_N)
    return Y.to(X.dtype)

# Example usage:
if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    X = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
    result = kernel(X)
    print(result)
