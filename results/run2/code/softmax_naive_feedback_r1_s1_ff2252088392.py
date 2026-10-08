import torch
import triton
import triton.language as tl
import math

# Triton kernel for row-wise softmax
@triton.jit
def softmax_kernel(
    X,  # input pointer
    Y,  # output pointer
    N,  # number of rows
    M,  # number of columns
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each row is processed by a separate block
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE_N) + row * M
    
    # Load data into registers
    col = tl.load(X + col_offsets, mask=col_offsets < M)
    
    # Compute max value in the row to avoid overflow
    max_val = tl.max(col, axis=0)
    col = col - max_val
    
    # Compute exp and sum of exps
    exp_col = tl.exp(col)
    sum_exp_col = tl.sum(exp_col, axis=0)
    
    # Normalize by sum of exps
    col = exp_col / sum_exp_col
    
    # Store the result
    tl.store(Y + col_offsets, col, mask=col_offsets < M)

# Wrapper function to call the Triton kernel
def kernel(X):
    N, M = X.shape
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    
    # Allocate output tensor
    Y = torch.empty_like(X)
    
    # Launch the Triton kernel
    grid = lambda META: (N,)
    softmax_kernel[grid](
        X, Y,
        N, M,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return Y

# Reference implementation for comparison
def reference(X):
    return torch.softmax(X.float(), dim=-1).to(X.dtype)

# Example usage
if __name__ == "__main__":
    # Create a random float16 tensor
    X = torch.rand((1024, 512), dtype=torch.float16, device='cuda')
    
    # Compute softmax using the reference and custom implementations
    Y_ref = reference(X)
    Y_custom = kernel(X)
    
    # Check if the results are the same within floating-point tolerance
    assert torch.allclose(Y_ref, Y_custom, atol=1e-3, rtol=1e-3)
    
    print("Results match!")
