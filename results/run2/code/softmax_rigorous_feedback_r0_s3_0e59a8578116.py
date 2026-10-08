import torch
import triton
import triton.language as tl

@triton.jit
def rowwise_softmax_kernel(X_ptr, Y_ptr, stride_xm, stride_ym, N, M, BLOCK_SIZE_M: tl.constexpr):
    # Load the row into shared memory
    row_idx = tl.program_id(axis=0)
    col_idx = tl.arange(0, BLOCK_SIZE_M)
    
    # Load the row from global memory to shared memory
    row = tl.load(X_ptr + row_idx * stride_xm + col_idx, mask=col_idx < M, other=-float('inf'))
    
    # Compute max and subtract it for numerical stability
    row_max = tl.max(row)
    row -= row_max
    
    # Compute exponentials
    exp_row = tl.exp(row)
    
    # Compute sum of exponentials
    sum_exp = tl.sum(exp_row)
    
    # Compute softmax
    softmax_row = exp_row / sum_exp
    
    # Store the result back to global memory
    tl.store(Y_ptr + row_idx * stride_ym + col_idx, softmax_row, mask=col_idx < M)

def kernel(X):
    assert X.is_cuda and X.dtype == torch.float16, "Input must be a CUDA float16 tensor"
    
    N, M = X.shape
    BLOCK_SIZE_M = 128
    
    # Allocate output tensor
    Y = torch.empty_like(X)
    
    # Define grid and block dimensions
    grid = lambda meta: (N,)
    block = lambda meta: (BLOCK_SIZE_M,)
    
    # Launch the Triton kernel
    rowwise_softmax_kernel[grid](X.data_ptr(), Y.data_ptr(), X.stride(0), Y.stride(0), N, M, BLOCK_SIZE_M)
    
    return Y

# Example usage
if __name__ == "__main__":
    import torch
    
    # Create a random float16 tensor
    X = torch.rand((1024, 128), dtype=torch.float16, device='cuda')
    
    # Compute softmax using the Triton kernel
    Y_triton = kernel(X)
    
    # Compute softmax using PyTorch's built-in function for comparison
    Y_pytorch = torch.softmax(X.float(), dim=-1).to(torch.float16)
    
    # Check if the results are close
    print(torch.allclose(Y_triton, Y_pytorch))
