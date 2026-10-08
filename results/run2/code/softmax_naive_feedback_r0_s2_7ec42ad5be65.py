import torch
import triton
import triton.language as tl
import math

# Triton kernel for row-wise softmax
@triton.jit
def softmax_kernel(
    X, Y, 
    BLOCK_M: tl.constexpr,
    ROW_SIZE: tl.constexpr
):
    # Get the row and column indices
    row = tl.program_id(0)
    col = tl.load(X + row * ROW_SIZE + tl.arange(0, BLOCK_M), mask=col < BLOCK_M)
    
    # Compute max and exp values
    max_val = tl.max(col, axis=0)
    exp_col = tl.exp(col - max_val)
    
    # Compute sum of exp values
    sum_exp = tl.sum(exp_col, axis=0)
    
    # Compute softmax values
    softmax_col = exp_col / sum_exp
    
    # Store the results
    tl.store(Y + row * ROW_SIZE + tl.arange(0, BLOCK_M), softmax_col, mask=col < BLOCK_M)

# Wrapper function to call the Triton kernel
def kernel(X):
    assert X.is_cuda and X.dtype == torch.float16, "Input must be a CUDA float16 tensor"
    
    M, N = X.shape
    BLOCK_M = 32
    GRID_M = (M + BLOCK_M - 1) // BLOCK_M
    
    # Allocate output tensor
    Y = torch.empty_like(X)
    
    # Launch the Triton kernel
    softmax_kernel[GRID_M](X, Y, BLOCK_M=BLOCK_M, ROW_SIZE=N)
    
    return Y

# Example usage
if __name__ == "__main__":
    import torch
    import triton
    
    # Create a random float16 tensor
    X = torch.rand((1024, 512), device='cuda', dtype=torch.float16)
    
    # Compute softmax using the Triton kernel
    Y_triton = kernel(X)
    
    # Compute softmax using PyTorch for comparison
    Y_torch = torch.softmax(X.float(), dim=-1).to(torch.float16)
    
    # Check if the results are close
    assert torch.allclose(Y_triton, Y_torch, atol=1e-3, rtol=1e-3), "Results do not match"
    
    print("Triton softmax matches PyTorch softmax")
