import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel for RMSNorm
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, N, M, BLOCK_SIZE: tl.constexpr):
    # Each thread computes one element of the output
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    row_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_idx = bid

    # Load the input data into shared memory
    x = tl.load(x_ptr + row_idx[:, None] + col_idx[None, :], mask=row_idx[:, None] < N, other=0.0)

    # Compute the mean of squares
    sum_squares = tl.sum(x * x, axis=0)
    mean_squares = sum_squares / N

    # Add epsilon to avoid division by zero
    epsilon = tl.full((BLOCK_SIZE,), 1e-6, dtype=tl.float32)
    denom = tl.sqrt(mean_squares + epsilon)

    # Normalize and scale by weight
    x_normalized = x / denom
    w = tl.load(w_ptr + col_idx, mask=col_idx < M, other=0.0)
    out = x_normalized * w

    # Store the result back to global memory
    tl.store(out_ptr + row_idx[:, None] + col_idx[None, :], out, mask=row_idx[:, None] < N)

# Triton kernel wrapper function
def kernel(x, w):
    assert x.dtype == torch.float16, "Input x must be of type torch.float16"
    assert w.dtype == torch.float16, "Weight w must be of type torch.float16"
    assert x.dim() == 2, "Input x must be a 2-D tensor"
    assert w.dim() == 1, "Weight w must be a 1-D tensor"
    assert x.size(1) == w.size(0), "The last dimension of x must match the size of w"

    N, M = x.size()
    BLOCK_SIZE = 128

    # Allocate output tensor
    out = torch.empty_like(x)

    # Launch the Triton kernel
    grid = lambda META: (triton.cdiv(N, BLOCK_SIZE), M)
    rmsnorm_kernel[grid](x, w, out, N, M, BLOCK_SIZE=BLOCK_SIZE)

    return out

# Example usage
if __name__ == "__main__":
    # Create random input tensors
    x = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
    w = torch.randn(512, device='cuda', dtype=torch.float16)

    # Run the Triton kernel
    result = kernel(x, w)

    # Print the result
    print(result)
