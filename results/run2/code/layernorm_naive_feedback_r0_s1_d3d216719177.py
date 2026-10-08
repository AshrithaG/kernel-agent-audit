import torch
import triton
import triton.language as tl

# Define the Triton kernel for LayerNorm
@triton.jit
def layer_norm_kernel(
    X, W, B, Y,
    stride_xz, stride_yz,
    N, D,
    BLOCK_SIZE: tl.constexpr
):
    # Get the thread ID
    idx = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    
    # Calculate the row and column indices
    row = idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = pid
    
    # Load the data into registers
    x = tl.load(X + row[:, None] * stride_xz + col * stride_yz, mask=row < N, other=0.0)
    
    # Compute mean and variance
    mean = tl.sum(x, axis=1) / D
    x_centered = x - mean[:, None]
    var = tl.sum(x_centered ** 2, axis=1) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    
    # Normalize and scale with weight and bias
    y = x_centered * rstd[:, None] * W[col] + B[col]
    
    # Store the result
    tl.store(Y + row[:, None] * stride_xz + col * stride_yz, y, mask=row < N)

# Define the kernel wrapper function
def kernel(X, W, B):
    # Ensure inputs are on the same device
    assert X.device == W.device == B.device, "All inputs must be on the same device"
    assert X.dtype == torch.float16, "Input tensor X must be of type float16"
    assert W.dtype == torch.float16, "Weight tensor W must be of type float16"
    assert B.dtype == torch.float16, "Bias tensor B must be of type float16"
    
    # Get the dimensions
    N, D = X.shape
    BLOCK_SIZE = 128
    
    # Allocate output tensor
    Y = torch.empty_like(X)
    
    # Launch the kernel
    grid = lambda META: (
        triton.cdiv(N, META['BLOCK_SIZE']),
        D
    )
    layer_norm_kernel[grid](
        X, W, B, Y,
        X.stride(0), Y.stride(0),
        N, D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return Y

# Example usage
if __name__ == "__main__":
    # Create example inputs
    X = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
    W = torch.randn(512, device='cuda', dtype=torch.float16)
    B = torch.randn(512, device='cuda', dtype=torch.float16)
    
    # Run the custom kernel
    Y_custom = kernel(X, W, B)
    
    # Run the reference implementation
    Y_ref = F.layer_norm(X.float(), (X.shape[-1],), W.float(), B.float(), 1e-5).to(X.dtype)
    
    # Check if the results are close
    print(torch.allclose(Y_custom, Y_ref))
