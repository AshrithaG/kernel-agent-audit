import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel
@triton.jit
def bias_gelu_kernel(
    x_ptr, b_ptr, out_ptr,
    X_BLOCK_SIZE: tl.constexpr,
    Y_BLOCK_SIZE: tl.constexpr,
    X_SIZE: tl.constexpr,
    Y_SIZE: tl.constexpr,
):
    # Get the thread ID
    x_id = tl.program_id(axis=0) * X_BLOCK_SIZE + tl.thread_idx(axis=0)
    y_id = tl.program_id(axis=1) * Y_BLOCK_SIZE + tl.thread_idx(axis=1)

    # Load the input values
    if x_id < X_SIZE and y_id < Y_SIZE:
        x_val = tl.load(x_ptr + y_id * X_SIZE + x_id, mask=x_id < X_SIZE, other=0.0)
        b_val = tl.load(b_ptr + y_id, mask=y_id < Y_SIZE, other=0.0)
        
        # Add bias
        x_plus_b = x_val + b_val
        
        # Compute tanh approximation of GELU
        k = 0.7978845608028654
        inner = 0.5 * (1.0 + tl.tanh(k * (x_plus_b + 0.044715 * (x_plus_b ** 3))))
        gelu_val = x_plus_b * inner
        
        # Store the result
        tl.store(out_ptr + y_id * X_SIZE + x_id, gelu_val, mask=x_id < X_SIZE)

# Define the wrapper function
def kernel(x, b):
    # Ensure the inputs are float16 and on the same device
    x = x.to(torch.float16)
    b = b.to(torch.float16)
    
    # Get the shape of the inputs
    X_SIZE, Y_SIZE = x.shape
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Define grid and block sizes
    X_BLOCK_SIZE = 128
    Y_BLOCK_SIZE = 32
    grid = (triton.cdiv(X_SIZE, X_BLOCK_SIZE), triton.cdiv(Y_SIZE, Y_BLOCK_SIZE))
    block = (X_BLOCK_SIZE, Y_BLOCK_SIZE)
    
    # Launch the kernel
    bias_gelu_kernel[grid](x, b, out, X_BLOCK_SIZE=X_BLOCK_SIZE, Y_BLOCK_SIZE=Y_BLOCK_SIZE, X_SIZE=X_SIZE, Y_SIZE=Y_SIZE)
    
    return out

# Example usage
if __name__ == "__main__":
    x = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
    b = torch.randn(512, device='cuda', dtype=torch.float16)
    result = kernel(x, b)
    print(result)
