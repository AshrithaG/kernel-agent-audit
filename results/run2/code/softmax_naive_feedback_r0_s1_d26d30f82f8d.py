import torch
import triton
import triton.language as tl
import math

# Triton kernel for row-wise softmax
@triton.jit
def softmax_kernel(
    X,  # input matrix
    Y,  # output matrix
    N,  # number of rows
    M,  # number of columns
    BLOCK_M: tl.constexpr,
):
    # Define the block size
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    block_start = pid * BLOCK_M + bid * M
    offsets_m = block_start + tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, M)
    mask_m = offsets_m < N
    mask_n = offsets_n < M
    # Load data from X
    x = tl.load(X + offsets_m[:, None] * M + offsets_n[None, :], mask=mask_m[:, None] & mask_n[None, :], other=-float('inf'))
    # Compute max of each row
    row_max = tl.max(x, axis=1)
    row_max = tl.broadcast_to(row_max[:, None], x.shape)
    # Subtract row max to avoid overflow
    x = x - row_max
    # Compute exp
    x_exp = tl.exp(x)
    # Compute sum of exp for each row
    row_sum = tl.sum(x_exp, axis=1)
    row_sum = tl.broadcast_to(row_sum[:, None], x.shape)
    # Compute softmax
    y = x_exp / row_sum
    # Store result to Y
    tl.store(Y + offsets_m[:, None] * M + offsets_n[None, :], y, mask=mask_m[:, None] & mask_n[None, :])

# Wrapper function to launch the Triton kernel
def kernel(X):
    N, M = X.shape
    # Define block size
    BLOCK_M = 128
    # Allocate output tensor
    Y = torch.empty_like(X)
    # Launch Triton kernel
    grid = lambda opt: (
        triton.cdiv(N, BLOCK_M),
        triton.cdiv(M, opt['BLOCK_N']),
    )
    softmax_kernel[grid](X, Y, N, M, BLOCK_M=BLOCK_M, BLOCK_N=M)
    return Y

# Reference implementation for comparison
def reference(X):
    return torch.softmax(X.float(), dim=-1).to(X.dtype)

# Example usage
if __name__ == "__main__":
    # Create a random input tensor
    X = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
    # Compute softmax using the Triton kernel
    Y_triton = kernel(X)
    # Compute softmax using the reference implementation
    Y_ref = reference(X)
    # Check if the results are close
    assert torch.allclose(Y_triton, Y_ref, atol=1e-3, rtol=1e-3)
    print("Triton kernel output matches the reference implementation.")
