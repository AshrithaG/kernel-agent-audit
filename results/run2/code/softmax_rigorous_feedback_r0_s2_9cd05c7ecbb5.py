import torch
import triton
import triton.language as tl
import math

@triton.jit
def softmax_kernel(
    X,  # pointer to input matrix
    Y,  # pointer to output matrix
    N,  # number of rows
    M,  # number of columns
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    # Each thread block computes one row of the output
    row_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    col_idx = bid * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load input values into shared memory
    x = tl.load(X + row_idx[:, None] * M + col_idx[None, :], mask=row_idx[:, None] < N & col_idx[None, :] < M, other=-float('inf'))

    # Compute max value along the row
    max_val = tl.max(x, axis=1)
    x = x - max_val[:, None]

    # Compute exp and sum of exps
    exp_x = tl.exp(x)
    sum_exp_x = tl.sum(exp_x, axis=1)

    # Compute softmax
    softmax_x = exp_x / sum_exp_x[:, None]

    # Store results back to global memory
    tl.store(Y + row_idx[:, None] * M + col_idx[None, :], softmax_x, mask=row_idx[:, None] < N & col_idx[None, :] < M)

def kernel(X):
    N, M = X.shape
    Y = torch.empty_like(X)
    
    # Define block sizes
    BLOCK_M = 128
    BLOCK_N = 32
    
    # Launch kernel
    grid = lambda META: (triton.cdiv(N, BLOCK_M), triton.cdiv(M, BLOCK_N))
    softmax_kernel[grid](X, Y, N, M, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    
    return Y

# Example usage
if __name__ == "__main__":
    X = torch.randn(1024, 128, device='cuda', dtype=torch.float16)
    Y_ref = torch.softmax(X, dim=-1)
    Y = kernel(X)
    print(torch.allclose(Y, Y_ref))
