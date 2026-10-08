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
    x_hat = (x - mean) / tl.sqrt(var + EPS)
    y = x_hat * tl.load(W + off_z * stride_xz + off_m * stride_xm, mask=off_m < M, other=0.0) + tl.load(B + off_z * stride_xz + off_m * stride_xm, mask=off_m < M, other=0.0)
    
    # Store the result
    tl.store(Y + off_z * stride_yz + off_m * stride_ym, y, mask=off_m < M)

# Define the kernel wrapper function
def kernel(X, W, B):
    # Ensure the input tensors are on the same device
    assert X.device == W.device == B.device, "All inputs must be on the same device"
    assert X.dtype == torch.float16, "Input tensor X must be of type torch.float16"
    assert W.dtype == torch.float16, "Weight tensor W must be of type torch.float16"
    assert B.dtype == torch.float16, "Bias tensor B must be of type torch.float16"
    
    # Get the shape of the input tensor
    N, M = X.shape
    
    # Create an output tensor of the same shape as X
    Y = torch.empty_like(X)
    
    # Define the grid and block sizes
    BLOCK_SIZE_M = 128
    grid = (triton.cdiv(M, BLOCK_SIZE_M), N)
    
    # Launch the kernel
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
    # Create random input tensors
    X = torch.randn(32, 1024, dtype=torch.float16, device='cuda')
    W = torch.randn(1024, dtype=torch.float16, device='cuda')
    B = torch.randn(1024, dtype=torch.float16, device='cuda')
    
    # Run the kernel
    Y = kernel(X, W, B)
    
    # Print the shape and dtype of the output tensor
    print("Output shape:", Y.shape)
    print("Output dtype:", Y.dtype)
