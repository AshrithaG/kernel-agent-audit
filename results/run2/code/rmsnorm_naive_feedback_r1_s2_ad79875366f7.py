import torch
import triton
import triton.language as tl
import math

@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, N, M, eps, BLOCK_SIZE: tl.constexpr):
    # Define the program ID
    pid = tl.program_id(axis=0)
    
    # Calculate the row and column indices
    row = pid // BLOCK_SIZE
    col = pid % BLOCK_SIZE
    
    # Load the data into shared memory
    x = tl.load(x_ptr + row * M + col, mask=(col < M), other=0.0)
    
    # Compute the sum of squares
    sum_squares = tl.sum(x**2, axis=1)
    
    # Compute the root mean square
    rms = tl.sqrt(sum_squares / M + eps)
    
    # Load the weight
    w = tl.load(w_ptr + col, mask=(col < M), other=0.0)
    
    # Normalize and scale
    normalized_x = x / rms
    scaled_x = normalized_x * w
    
    # Store the result
    tl.store(out_ptr + row * M + col, scaled_x, mask=(col < M))

def kernel(x, w):
    assert x.is_cuda and w.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == torch.float16 and w.dtype == torch.float16, "Inputs must be float16 tensors"
    assert x.ndim == 2 and w.ndim == 1 and x.shape[1] == w.shape[0], "Invalid input shapes"
    
    N, M = x.shape
    eps = 1e-6
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Define the grid
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    # Launch the kernel
    rmsnorm_kernel[grid](x, w, out, N, M, eps, BLOCK_SIZE=128)
    
    return out

# Example usage
if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    x = torch.randn(32, 1024, device='cuda', dtype=torch.float16)
    w = torch.randn(1024, device='cuda', dtype=torch.float16)
    
    out = kernel(x, w)
    print(out)
