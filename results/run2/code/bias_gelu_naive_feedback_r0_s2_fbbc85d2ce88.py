import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel
@triton.jit
def add_bias_kernel(X_ptr, B_ptr, Y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    # Define the indices
    row = tl.program_id(0)  # Batch index
    col = tl.load(B_ptr + row)  # Load the bias value for this row
    x = tl.load(X_ptr + row * N + col)  # Load the corresponding element from X
    y = x + col  # Add the bias
    # Approximate GELU using tanh
    k = 0.7978845608028654 * y
    y = 0.5 * y * (1.0 + tl.tanh(k))
    tl.store(Y_ptr + row * N + col, y)  # Store the result

# Define the wrapper function
def kernel(X, B):
    # Get the shapes
    M, N = X.shape
    # Allocate the output tensor
    Y = torch.empty_like(X)
    # Define the grid
    grid = lambda META: (M,)
    # Launch the kernel
    add_bias_kernel[grid](X, B, Y, M, N, BLOCK_SIZE=128)
    return Y

# Example usage
if __name__ == "__main__":
    # Create example inputs
    X = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    B = torch.randn(1024, device='cuda', dtype=torch.float16)
    # Call the kernel
    Y = kernel(X, B)
    print(Y)
