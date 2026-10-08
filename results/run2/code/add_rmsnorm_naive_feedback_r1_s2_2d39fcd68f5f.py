import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel for RMSNorm
@triton.jit
def rmsnorm_kernel(
    x_ptr, r_ptr, w_ptr, out_ptr, 
    N: tl.constexpr, M: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the row index for the current thread block
    row = tl.program_id(axis=0)
    # Calculate the column index for the current thread block
    col = tl.arange(0, BLOCK_SIZE, dtype=tl.int32)
    
    # Load the input data into shared memory
    x = tl.load(x_ptr + row * M + col, mask=(col < M), other=0.0)
    r = tl.load(r_ptr + row * M + col, mask=(col < M), other=0.0)
    
    # Add residual
    h = x + r
    
    # Compute the mean of squares
    mean_squares = tl.sum(h * h, axis=1)
    
    # Compute the reciprocal square root of the mean of squares plus epsilon
    rsqrt_mean_squares = 1.0 / tl.sqrt(mean_squares + 1e-6)
    
    # Load the weights
    w = tl.load(w_ptr + col, mask=(col < M), other=0.0)
    
    # Scale the result by the weights
    out = h * rsqrt_mean_squares * w
    
    # Store the result back to global memory
    tl.store(out_ptr + row * M + col, out, mask=(col < M))

# Triton kernel wrapper function
def kernel(x, r, w):
    N, M = x.shape
    # Define the block size
    BLOCK_SIZE = 128
    # Launch the Triton kernel
    rmsnorm_kernel[(N,)](x, r, w, x, N, M, BLOCK_SIZE=BLOCK_SIZE)
    return x

# Example usage
if __name__ == "__main__":
    # Create example inputs
    x = torch.randn(32, 1024, dtype=torch.float16, device='cuda')
    r = torch.randn(32, 1024, dtype=torch.float16, device='cuda')
    w = torch.randn(1024, dtype=torch.float16, device='cuda')
    
    # Run the Triton kernel
    output = kernel(x, r, w)
    
    # Print the shape and dtype of the output
    print(output.shape, output.dtype)
