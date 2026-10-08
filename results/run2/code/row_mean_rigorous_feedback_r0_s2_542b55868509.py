import torch
import triton
import triton.language as tl
import math

# Triton kernel to compute mean over the last dimension of a 2-D tensor
@triton.jit
def mean_kernel(
    X,  # input matrix
    Y,  # output vector
    M,  # number of rows
    N,  # number of columns
    BLOCK_SIZE: tl.constexpr,  # number of elements processed by each thread
):
    # Each thread computes the sum of a block of elements in its row
    pid = tl.program_id(axis=0)  # get the id of the current row
    if pid >= M:
        return
    row_start = pid * BLOCK_SIZE
    row_end = min(row_start + BLOCK_SIZE, N)
    row_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(row_start, row_end):
        row_sum[i - row_start] = X[pid, i]
    row_sum = tl.sum(row_sum)
    # Compute the mean by dividing the sum by the number of elements
    row_mean = row_sum / (row_end - row_start)
    Y[pid] = row_mean

# Triton wrapper function
def kernel(X):
    M, N = X.shape
    # Define the block size
    BLOCK_SIZE = 128
    # Allocate output tensor
    Y = torch.empty((M,), dtype=torch.float32, device=X.device)
    # Launch the Triton kernel
    grid = lambda META: (triton.cdiv(M, META.BLOCK_SIZE),)
    mean_kernel[grid](X, Y, M, N, BLOCK_SIZE=BLOCK_SIZE)
    # Convert the result to the original dtype
    return Y.to(X.dtype)

# Example usage
if __name__ == "__main__":
    # Create a random 2-D tensor of shape (M, N) with float16 values
    M, N = 1024, 512
    X = torch.rand((M, N), dtype=torch.float16, device='cuda')
    
    # Compute the mean using the Triton kernel
    Y_triton = kernel(X)
    
    # Compute the mean using the reference implementation
    Y_ref = X.float().mean(dim=-1).to(X.dtype)
    
    # Check if the results are the same within floating-point tolerance
    assert torch.allclose(Y_triton, Y_ref), "The results do not match!"
    
    print("Results match!")
