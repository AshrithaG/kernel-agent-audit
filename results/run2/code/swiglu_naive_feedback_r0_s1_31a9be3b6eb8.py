import torch
import triton
import triton.language as tl
import math

# Triton kernel for SiLU activation
@triton.jit
def silu_kernel(x_ptr, y_ptr, N: tl.constexpr):
    # Get the thread index
    idx = tl.program_id(axis=0) * tl.num_threads() + tl.thread_idx()
    # Check if the index is within bounds
    if idx < N:
        # Load the value from the input tensor
        x_val = tl.load(x_ptr + idx)
        # Apply the SiLU activation function
        y_val = x_val / (1.0 + tl.exp(-x_val))
        # Store the result in the output tensor
        tl.store(y_ptr + idx, y_val)

# Triton kernel for SwiGLU operation
@triton.jit
def swiglu_kernel(x_ptr, y_ptr, N: tl.constexpr, M: tl.constexpr):
    # Get the thread index
    idx = tl.program_id(axis=0) * tl.num_threads() + tl.thread_idx()
    # Calculate the index for the first half and second half
    idx1 = idx // M
    idx2 = idx % M
    # Check if the index is within bounds
    if idx1 < N:
        # Load the values from the input tensor
        x1_val = tl.load(x_ptr + idx1 * M + idx2)
        x2_val = tl.load(x_ptr + idx1 * M + M // 2 + idx2)
        # Apply the SiLU activation function to the first half
        y1_val = x1_val / (1.0 + tl.exp(-x1_val))
        # Multiply the results
        y_val = y1_val * x2_val
        # Store the result in the output tensor
        tl.store(y_ptr + idx1 * M + idx2, y_val)

# Wrapper function for SwiGLU using Triton
def kernel(x):
    N = x.shape[0]
    M = x.shape[-1] // 2
    y = torch.empty_like(x)
    
    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(N * M, meta['BLOCK_SIZE']),)
    swiglu_kernel[grid](x.contiguous(), y.contiguous(), N, M, BLOCK_SIZE=1024)
    
    return y

# Example usage
if __name__ == "__main__":
    # Create a random float16 tensor
    x = torch.rand((1024, 1024), dtype=torch.float16, device='cuda')
    
    # Compute SwiGLU using the Triton kernel
    y = kernel(x)
    
    # Print the shape and dtype of the result
    print("Shape:", y.shape)
    print("Dtype:", y.dtype)
