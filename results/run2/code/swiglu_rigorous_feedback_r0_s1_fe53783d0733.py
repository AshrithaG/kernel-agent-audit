import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel for SiLU activation
@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the index of the current thread
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.thread_idx(axis=0)
    
    # Load the input value
    x = tl.load(x_ptr + idx, mask=idx < n_elements, other=0.0)
    
    # Apply the SiLU activation function
    y = x * tl.sigmoid(x)
    
    # Store the result
    tl.store(y_ptr + idx, y, mask=idx < n_elements)

# Define the Triton kernel for element-wise multiplication
@triton.jit
def mul_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the index of the current thread
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.thread_idx(axis=0)
    
    # Load the input values
    x = tl.load(x_ptr + idx, mask=idx < n_elements, other=0.0)
    y = tl.load(y_ptr + idx, mask=idx < n_elements, other=0.0)
    
    # Perform element-wise multiplication
    z = x * y
    
    # Store the result
    tl.store(z_ptr + idx, z, mask=idx < n_elements)

# Define the main function to execute the Triton kernels
def kernel(x):
    # Get the shape of the input tensor
    batch_size, seq_len, n_features = x.shape
    
    # Split the last dimension into two halves
    n_half_features = n_features // 2
    
    # Create output tensors
    x1 = x[..., :n_half_features]
    x2 = x[..., n_half_features:]
    
    # Allocate memory for the SiLU results
    y1 = torch.empty_like(x1)
    
    # Define the grid and block sizes for the SiLU kernel
    BLOCK_SIZE = 1024
    num_blocks = math.ceil(n_half_features * seq_len * batch_size / BLOCK_SIZE)
    
    # Launch the SiLU kernel
    silu_kernel[(num_blocks,)](x1.contiguous(), y1.contiguous(), n_half_features * seq_len * batch_size, BLOCK_SIZE=BLOCK_SIZE)
    
    # Allocate memory for the final output
    out = torch.empty_like(x)
    
    # Launch the multiplication kernel
    mul_kernel[(num_blocks,)](y1.contiguous(), x2.contiguous(), out.contiguous(), n_half_features * seq_len * batch_size, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

# Example usage
if __name__ == "__main__":
    # Create a random input tensor
    x = torch.randn(8, 128, 512, device='cuda', dtype=torch.float16)
    
    # Apply the Triton-based SiGLU function
    result = kernel(x)
    
    print(result.shape)
    print(result.dtype)
