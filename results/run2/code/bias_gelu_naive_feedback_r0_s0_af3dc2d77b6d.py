import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel
@triton.jit
def gelu_tanh_kernel(
    X, B, Y,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # Get the thread ID
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    # Calculate the row and column indices
    row = pid * BLOCK_M
    col = bid * BLOCK_N

    # Load the data into registers
    x = tl.load(X + row * 128 + col, mask=row < tl.num_programs(axis=0) & col < tl.num_programs(axis=1))
    b = tl.load(B + col, mask=col < tl.num_programs(axis=1))

    # Add bias to the row
    x += b

    # Compute the GELU approximation using tanh
    cdf = 0.5 * (1.0 + tl.tanh((x / 1.4142135623730951) * (1.0 + 0.044715 * (x ** 2))))
    y = x * cdf

    # Store the result back to global memory
    tl.store(Y + row * 128 + col, y, mask=row < tl.num_programs(axis=0) & col < tl.num_programs(axis=1))

# Define the kernel wrapper
def kernel(X, B):
    # Ensure the input tensors are on the same device
    assert X.device == B.device, "Inputs must be on the same device"
    assert X.dtype == torch.float16, "Input X must be of type float16"
    assert B.dtype == torch.float16, "Input B must be of type float16"

    # Get the shapes of the input tensors
    M, N = X.shape

    # Allocate output tensor
    Y = torch.empty_like(X)

    # Define the grid and block sizes
    BLOCK_M = 32
    BLOCK_N = 128
    grid = (tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N))

    # Launch the kernel
    gelu_tanh_kernel[grid](X, B, Y, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    return Y

# Example usage
if __name__ == "__main__":
    # Create example input tensors
    X = torch.rand(1024, 128, device='cuda', dtype=torch.float16)
    B = torch.rand(128, device='cuda', dtype=torch.float16)

    # Run the kernel
    Y = kernel(X, B)

    # Print the result
    print(Y)
