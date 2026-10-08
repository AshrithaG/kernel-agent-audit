import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel for adding bias and approximating GELU
@triton.jit
def add_bias_gelu_kernel(
    x_ptr, b_ptr, y_ptr, 
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # Get the thread ID
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    
    # Calculate the row and column indices
    row = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    col = bid * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Load the elements from the input matrices
    x = tl.load(x_ptr + row[:, None] * N + col[None, :], mask=row[:, None] < M & col[None, :] < N, other=0.0)
    b = tl.load(b_ptr + col, mask=col < N, other=0.0)
    
    # Add bias to each element
    x = x + b
    
    # Approximate GELU using tanh
    cdf = 0.5 * (1.0 + tl.tanh((math.sqrt(2.0 / math.pi) * (x + 0.044715 * tl.pow(x, 3)))))
    y = x * cdf
    
    # Store the result back to the output matrix
    tl.store(y_ptr + row[:, None] * N + col[None, :], y, mask=row[:, None] < M & col[None, :] < N)

# Define the wrapper function to call the Triton kernel
def kernel(x, b):
    # Ensure the input tensors are on the same device
    assert x.device == b.device, "Inputs must be on the same device"
    
    # Get the shape of the input tensors
    M, N = x.shape
    
    # Define the block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    
    # Calculate the grid size
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    
    # Allocate the output tensor
    y = torch.empty_like(x)
    
    # Launch the Triton kernel
    add_bias_gelu_kernel[(grid_m, grid_n)](
        x.contiguous(), b.contiguous(), y.contiguous(),
        M, N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )
    
    return y

# Example usage
if __name__ == "__main__":
    # Create example input tensors
    x = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
    b = torch.randn(512, dtype=torch.float16, device='cuda')
    
    # Call the Triton kernel
    y = kernel(x, b)
    
    # Print the result
    print(y)
