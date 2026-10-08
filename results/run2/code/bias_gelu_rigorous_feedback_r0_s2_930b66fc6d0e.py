import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel for adding bias and applying GELU approximation
@triton.jit
def bias_gelu_kernel(
    X, B, Y,
    X.stride(0), X.stride(1),
    B.stride(0), B.stride(1),
    Y.stride(0), Y.stride(1),
    N: tl.constexpr, M: tl.constexpr
):
    # Get the row index
    row = tl.program_id(0)
    # Load the bias vector into shared memory
    b = tl.load(B + row * B.stride(0), mask=row < N, other=0.0)
    # Initialize the output row
    y_row = tl.zeros([M], dtype=tl.float32)
    # Iterate over columns
    for col in range(M):
        # Load the element from the input matrix
        x = tl.load(X + row * X.stride(0) + col * X.stride(1), mask=(row < N) & (col < M), other=0.0)
        # Add bias
        x += b
        # Apply GELU approximation using tanh
        x = 0.5 * x * (1 + tl.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
        # Store the result in the output row
        y_row[col] = x
    # Write the output row to global memory
    tl.store(Y + row * Y.stride(0), y_row, mask=row < N)

def kernel(X, B):
    # Ensure inputs are float16
    X = X.to(torch.float16)
    B = B.to(torch.float16)
    
    # Get the shape of the input tensor
    N, M = X.shape
    
    # Allocate output tensor
    Y = torch.empty_like(X)
    
    # Define grid and block dimensions
    BLOCK_SIZE = 128
    grid = lambda opt: (N, )
    block = (BLOCK_SIZE, )
    
    # Launch the Triton kernel
    bias_gelu_kernel[grid](X, B, Y, X.stride(0), X.stride(1), B.stride(0), B.stride(1), Y.stride(0), Y.stride(1), N, M)
    
    return Y

# Example usage
if __name__ == "__main__":
    # Create example inputs
    X = torch.rand((1024, 512), device='cuda', dtype=torch.float16)
    B = torch.rand((1024,), device='cuda', dtype=torch.float16)
    
    # Run the kernel
    Y = kernel(X, B)
    
    print(Y)
