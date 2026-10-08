import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel
@triton.jit
def add_bias_tanh_gelu_kernel(
    X, B, Y,
    X.stride(0), X.stride(1),
    B.stride(0),
    Y.stride(0), Y.stride(1),
    N: tl.constexpr, M: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    # Calculate the row index
    row = pid // M
    # Calculate the column index
    col = pid % M
    # Load the bias value for the current row
    bias = tl.load(B + row * B.stride(0), mask=(col < M))
    # Load the input value for the current row and column
    x = tl.load(X + row * X.stride(0) + col * X.stride(1), mask=(col < M))
    # Add the bias to the input value
    x_with_bias = x + bias
    # Apply the tanh approximation of GELU
    k = 0.7978845608028654 * x_with_bias
    tanh_k = tl.tanh(k)
    gelu_approx = 0.5 * x_with_bias * (1.0 + tanh_k)
    # Store the result in the output tensor
    tl.store(Y + row * Y.stride(0) + col * Y.stride(1), gelu_approx, mask=(col < M))

# Define the wrapper function
def kernel(X, B):
    # Get the shape of the input tensor
    N, M = X.shape
    # Allocate the output tensor
    Y = torch.empty_like(X)
    # Define the grid size
    grid_size = (triton.cdiv(N * M, 1024),)
    # Launch the Triton kernel
    add_bias_tanh_gelu_kernel[grid_size](
        X, B, Y,
        X.stride(0), X.stride(1),
        B.stride(0),
        Y.stride(0), Y.stride(1),
        N=N, M=M,
    )
    return Y

# Example usage
if __name__ == "__main__":
    # Create random input tensors
    X = torch.randn(128, 256, dtype=torch.float16, device="cuda")
    B = torch.randn(128, dtype=torch.float16, device="cuda")
    # Call the kernel function
    Y = kernel(X, B)
    # Print the shape and dtype of the output tensor
    print("Output shape:", Y.shape)
    print("Output dtype:", Y.dtype)
