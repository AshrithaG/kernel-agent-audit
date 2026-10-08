import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel for RMSNorm
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_elements, n_features, BLOCK_SIZE: tl.constexpr):
    # Get the thread ID
    pid = tl.program_id(axis=0)
    # Calculate the row index
    row_idx = tl.arange(0, BLOCK_SIZE)
    # Load the data into registers
    x = tl.load(x_ptr + row_idx[:, None] * n_features + pid, mask=row_idx[:, None] < n_features, other=0.0)
    w = tl.load(w_ptr + row_idx, mask=row_idx < n_features, other=0.0)
    # Compute the square of each element
    x_sq = x * x
    # Sum the squares along the features dimension
    sum_x_sq = tl.sum(x_sq, axis=1)
    # Add epsilon to avoid division by zero
    sum_x_sq += 1e-6
    # Compute the reciprocal of the square root of the sum
    rsqrt_sum_x_sq = 1.0 / tl.sqrt(sum_x_sq)
    # Scale the input by the reciprocal of the square root
    scaled_x = x * rsqrt_sum_x_sq[:, None]
    # Multiply by the weight vector
    weighted_x = scaled_x * w
    # Store the result back to global memory
    tl.store(out_ptr + row_idx[:, None] * n_features + pid, weighted_x, mask=row_idx[:, None] < n_features)

# Define the kernel wrapper function
def kernel(x, w):
    # Get the shape of the input tensor
    n_elements, n_features = x.shape
    # Allocate output tensor
    out = torch.empty_like(x)
    # Define the grid and block sizes
    grid = lambda opt: (n_elements, )
    block = 1024
    # Launch the Triton kernel
    rmsnorm_kernel[grid](x, w, out, n_elements, n_features, BLOCK_SIZE=block)
    return out

# Example usage
if __name__ == "__main__":
    # Create example input tensors
    x = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
    w = torch.randn(512, device='cuda', dtype=torch.float16)
    # Call the kernel function
    result = kernel(x, w)
    print(result)
