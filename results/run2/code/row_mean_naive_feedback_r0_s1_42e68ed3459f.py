import torch
import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def mean_kernel(
    X,  # input matrix
    Y,  # output vector
    N,  # number of rows
    M,  # number of columns
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread processes one row
    pid = tl.program_id(axis=0)
    if pid >= N:
        return
    
    # Load the entire row into shared memory
    row_ptr = X + pid * M
    row = tl.load(row_ptr, mask=tl.arange(0, M) < M)
    
    # Compute the sum of the row
    row_sum = tl.sum(row)
    
    # Store the result in the output vector
    Y[pid] = row_sum / M

# Define the wrapper function to call the Triton kernel
def kernel(X):
    N, M = X.shape
    Y = torch.empty(N, dtype=torch.float32, device=X.device)
    
    # Launch the Triton kernel
    grid = lambda META: (N,)
    mean_kernel[grid](X, Y, N, M, BLOCK_SIZE=1024)
    
    return Y.to(X.dtype)

# Example usage
if __name__ == "__main__":
    import torch
    X = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
    Y = kernel(X)
    print(Y)
