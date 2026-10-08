import torch
import triton
import triton.language as tl

# Define the Triton kernel for RMSNorm
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n, m, BLOCK_SIZE: tl.constexpr):
    # Each thread handles a single element in the output
    pid = tl.program_id(axis=0)
    row_id = pid // BLOCK_SIZE
    col_id = pid % BLOCK_SIZE
    
    # Load the data into registers
    if row_id < n and col_id < m:
        x = tl.load(x_ptr + row_id * m + col_id, mask=(col_id < m), other=0.0)
        w = tl.load(w_ptr + col_id, mask=(col_id < m), other=0.0)
        
        # Compute the mean of squares
        sum_squares = tl.sum(x * x, axis=1)
        mean_squares = sum_squares / m
        
        # Compute the RMS and normalize
        rms = tl.sqrt(mean_squares + 1e-6)
        normalized_x = x / rms
        
        # Apply the weight
        weighted_x = normalized_x * w
        
        # Store the result
        tl.store(out_ptr + row_id * m + col_id, weighted_x, mask=(col_id < m))

# Define the wrapper function to call the Triton kernel
def kernel(x, w):
    n, m = x.shape
    out = torch.empty_like(x)
    
    # Define the grid size
    grid_size = triton.cdiv(n * m, 1024)
    
    # Launch the Triton kernel
    rmsnorm_kernel[grid_size](x, w, out, n, m, BLOCK_SIZE=1024)
    
    return out

# Example usage
if __name__ == "__main__":
    x = torch.randn((128, 512), device='cuda', dtype=torch.float16)
    w = torch.randn(512, device='cuda', dtype=torch.float16)
    
    result = kernel(x, w)
    print(result)
