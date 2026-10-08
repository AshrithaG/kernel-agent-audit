import torch
import triton
import triton.language as tl

@triton.jit
def rowwise_softmax_kernel(X, Y, BLOCK_SIZE: tl.constexpr):
    # Get the position of the current thread within the block
    row = tl.program_id(0)
    col = tl.load(X + row * BLOCK_SIZE + tl threadIdx.x)
    
    # Compute max and subtract it from each element to avoid overflow
    max_val = tl.max(col, axis=0)
    col -= max_val
    
    # Compute exp and sum of exps
    exp_col = tl.exp(col)
    sum_exp_col = tl.sum(exp_col, axis=0)
    
    # Compute softmax
    softmax_col = exp_col / sum_exp_col
    
    # Store the result back to global memory
    tl.store(Y + row * BLOCK_SIZE + tl threadIdx.x, softmax_col)

def kernel(X):
    # Ensure X is a 2D tensor
    assert X.ndim == 2, "Input must be a 2D tensor"
    
    # Get the shape of the input tensor
    M, N = X.shape
    
    # Define the block size
    BLOCK_SIZE = 128
    
    # Allocate output tensor
    Y = torch.empty_like(X)
    
    # Launch the Triton kernel
    grid = lambda META: (M,)
    rowwise_softmax_kernel[grid](X, Y, BLOCK_SIZE=BLOCK_SIZE)
    
    return Y

# Example usage
if __name__ == "__main__":
    import torch
    
    # Create a random 2D tensor of shape (M, N) with float16 data type
    M, N = 1024, 128
    X = torch.rand(M, N, device='cuda', dtype=torch.float16)
    
    # Compute softmax using the Triton kernel
    Y = kernel(X)
    
    # Compute softmax using PyTorch for comparison
    Z = torch.softmax(X.float(), dim=-1).to(X.dtype)
    
    # Check if the results are the same within floating-point tolerance
    assert torch.allclose(Y, Z), "The results are not the same"
    
    print("Row-wise softmax computation is correct.")
