import torch
import triton
import triton.language as tl
import math

@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    
    # Calculate the row and column indices
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = bid
    
    # Load data from global memory to shared memory
    x = tl.load(x_ptr + row[:, None] + col[None, :], mask=row[:, None] < n, other=0.0)
    
    # Compute the square of each element
    x_sq = x * x
    
    # Sum the squares along the row
    sum_x_sq = tl.sum(x_sq, axis=0)
    
    # Compute the mean of the squares
    mean_x_sq = sum_x_sq / n
    
    # Add epsilon to avoid division by zero
    mean_x_sq += 1e-6
    
    # Compute the reciprocal square root
    rsqrt_mean_x_sq = 1.0 / tl.sqrt(mean_x_sq)
    
    # Load the weight vector
    w = tl.load(w_ptr + col, mask=col < m, other=0.0)
    
    # Normalize the elements and scale by the weight
    normalized_x = x * rsqrt_mean_x_sq * w
    
    # Store the result back to global memory
    tl.store(out_ptr + row[:, None] + col[None, :], normalized_x, mask=row[:, None] < n)

def kernel(x, w):
    n, m = x.shape
    BLOCK_SIZE = 128
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Launch the Triton kernel
    grid = (triton.cdiv(n, BLOCK_SIZE), m)
    rmsnorm_kernel[grid](x, w, out, n, m, BLOCK_SIZE)
    
    return out

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
    w = torch.randn(512, device='cuda', dtype=torch.float16)
    
    out = kernel(x, w)
    print(out)
