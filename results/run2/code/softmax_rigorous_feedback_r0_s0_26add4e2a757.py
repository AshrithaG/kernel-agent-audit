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
    BLOCK_M: tl.constexpr,  # number of columns per block
    **META
):
    # Each thread processes one element of the softmax
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    
    # Compute row and column indices
    row_idx = pid
    col_idx = bid * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Load data from global memory into shared memory
    x = tl.load(X + row_idx * M + col_idx, mask=col_idx < M, other=-float('inf'))
    
    # Compute max and subtract for numerical stability
    max_x = tl.max(x, axis=0)
    x = x - max_x
    
    # Compute exponential and sum of exponentials
    exp_x = tl.exp(x)
    sum_exp_x = tl.sum(exp_x, axis=0)
    
    # Compute softmax
    softmax_x = exp_x / sum_exp_x
    
    # Store result back to global memory
    tl.store(Y + row_idx * M + col_idx, softmax_x, mask=col_idx < M)

# Triton wrapper function
def kernel(X):
    N, M = X.shape
    assert X.is_cuda and X.dtype == torch.float16, "Input must be a CUDA float16 tensor"
    
    # Allocate output tensor
    Y = torch.empty_like(X)
    
    # Define grid and block sizes
    BLOCK_M = 128
    grid = (N, triton.cdiv(M, BLOCK_M))
    
    # Launch kernel
    softmax_kernel[grid](X, Y, N, M, BLOCK_M)
    
    return Y

# Reference implementation for testing
def reference(X):
    return torch.softmax(X.float(), dim=-1).to(X.dtype)

# Example usage
if __name__ == "__main__":
    import time
    
    # Test with random input
    X = torch.rand((1024, 512), device='cuda', dtype=torch.float16)
    
    # Measure reference time
    start_time = time.time()
    Y_ref = reference(X)
    ref_time = time.time() - start_time
    print(f"Reference time: {ref_time:.4f} seconds")
    
    # Measure Triton time
    start_time = time.time()
    Y_triton = kernel(X)
    triton_time = time.time() - start_time
    print(f"Triton time: {triton_time:.4f} seconds")
    
    # Check correctness
    assert torch.allclose(Y_ref, Y_triton, atol=1e-3, rtol=1e-3), "Output mismatch"
    print("Output is correct")
