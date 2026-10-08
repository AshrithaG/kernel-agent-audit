import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel for LayerNorm
@triton.jit
def layernorm_kernel(
    X, W, B, Y, 
    stride_xz, stride_xm, stride_yz, stride_ym,
    N, M,
    BLOCK_SIZE_M: tl.constexpr,
    EPS: tl.constexpr
):
    # Determine the thread ID
    off_m = tl.program_id(0) * BLOCK_SIZE_M + tl.thread_idx(0)
    off_z = tl.program_id(1)
    
    # Load the data into shared memory
    x = tl.load(X + off_z * stride_xz + off_m * stride_xm, mask=off_m < M, other=0.0)
    
    # Compute the mean and variance
    mean = tl.sum(x, axis=0) / N
    var = tl.sum((x - mean) ** 2, axis=0) / N
    
    # Normalize and scale
    norm_x = (x - mean) / tl.sqrt(var + EPS)
    y = norm_x * tl.load(W + off_z * stride_xz + off_m * stride_xm, mask=off_m < M, other=0.0) + tl.load(B + off_z * stride_xz + off_m * stride_xm, mask=off_m < M, other=0.0)
    
    # Store the result
    tl.store(Y + off_z * stride_yz + off_m * stride_ym, y, mask=off_m < M)

def kernel(X, W, B):
    # Get the shapes of the input tensors
    N, M = X.shape
    
    # Define the block size
    BLOCK_SIZE_M = 32
    
    # Define the grid
    grid = (triton.cdiv(M, BLOCK_SIZE_M), N)
    
    # Allocate the output tensor
    Y = torch.empty_like(X)
    
    # Launch the Triton kernel
    layernorm_kernel[grid](
        X, W, B, Y,
        X.stride(0), X.stride(1), Y.stride(0), Y.stride(1),
        N, M,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        EPS=1e-5
    )
    
    return Y

# Example usage
if __name__ == "__main__":
    # Create some example input tensors
    X = torch.randn(128, 512, device='cuda', dtype=torch.float16)
    W = torch.randn(512, device='cuda', dtype=torch.float16)
    B = torch.randn(512, device='cuda', dtype=torch.float16)
    
    # Run the custom kernel
    Y = kernel(X, W, B)
    
    # Print the result
    print(Y)
