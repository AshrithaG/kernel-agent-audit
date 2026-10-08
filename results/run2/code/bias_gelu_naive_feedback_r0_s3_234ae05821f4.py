import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel
@triton.jit
def add_bias_kernel(X_ptr, B_ptr, Y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    # Define the row and column indices
    row = tl.program_id(0)
    col = tl.load(B_ptr)  # Load the bias value for this row
    X_block_ptr = X_ptr + row * N
    Y_block_ptr = Y_ptr + row * N
    
    # Load the elements of the row from X into shared memory
    offsets = col + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X_block_ptr + offsets, mask=mask)
    
    # Apply the bias
    x += col
    
    # Approximate GELU using tanh
    cdf = 0.5 * (1.0 + tl.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
    y = x * cdf
    
    # Store the result back to Y
    tl.store(Y_block_ptr + offsets, y, mask=mask)

# Define the kernel wrapper
def kernel(X, B):
    assert X.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    assert X.dtype == torch.float16 and B.dtype == torch.float16, "Inputs must be float16"
    M, N = X.shape
    BLOCK_SIZE = 128  # Adjust block size based on performance testing
    
    # Allocate output tensor
    Y = torch.empty_like(X)
    
    # Launch the kernel
    grid = lambda META: (M,)
    add_bias_kernel[grid](X.data_ptr(), B.data_ptr(), Y.data_ptr(), M, N, BLOCK_SIZE)
    
    return Y

# Example usage
if __name__ == "__main__":
    # Create example inputs
    X = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
    B = torch.randn(1024, device='cuda', dtype=torch.float16)
    
    # Run the kernel
    Y = kernel(X, B)
    
    # Print the result
    print(Y)
