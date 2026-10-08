import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, N, M, eps, BLOCK_SIZE: tl.constexpr):
    # Define the thread indices
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    x_offset = bid * BLOCK_SIZE + pid
    w_offset = pid

    # Load the data into shared memory
    x = tl.load(x_ptr + x_offset, mask=x_offset < N, other=0.0)
    w = tl.load(w_ptr + w_offset, mask=pid < M, other=0.0)

    # Compute the mean of squares
    sum_squares = tl.sum(x * x, axis=0)
    mean_squares = sum_squares / N

    # Compute the root mean square with epsilon
    rms = tl.sqrt(mean_squares + eps)

    # Normalize and scale by weight
    normalized_x = x / rms
    scaled_x = normalized_x * w

    # Store the result
    tl.store(out_ptr + x_offset, scaled_x, mask=x_offset < N)

# Define the wrapper function
def kernel(x, w):
    N, M = x.shape
    assert M == w.shape[0], "The second dimension of x must match the length of w"

    # Allocate output tensor
    out = torch.empty_like(x)

    # Define grid and block dimensions
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE), M)
    block = (BLOCK_SIZE,)

    # Launch the kernel
    rmsnorm_kernel[grid](x, w, out, N, M, 1e-6, BLOCK_SIZE)

    return out

# Example usage
if __name__ == "__main__":
    import torch

    # Create example inputs
    x = torch.randn(1024, 128, device='cuda', dtype=torch.float16)
    w = torch.randn(128, device='cuda', dtype=torch.float16)

    # Compute RMSNorm using the Triton kernel
    result = kernel(x, w)
    print(result)
