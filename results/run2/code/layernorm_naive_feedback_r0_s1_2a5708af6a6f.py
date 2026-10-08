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
    
    # Apply weight and bias
    y = x_centered * rstd[:, None] * W[col] + B[col]
    
    # Store the result
    tl.store(Y + row[:, None] * stride_xz + col * stride_yz, y, mask=row < N)

# Define the kernel wrapper function
def kernel(X, W, B):
    # Ensure the input is a float16 tensor
    X = X.to(torch.float16)
    W = W.to(torch.float16)
    B = B.to(torch.float16)
    
    # Get the dimensions
    N, D = X.shape
    
    # Allocate output tensor
    Y = torch.empty_like(X)
    
    # Define grid and block sizes
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE), D)
    
    # Launch the kernel
    layer_norm_kernel[grid](
        X.data_ptr(), W.data_ptr(), B.data_ptr(), Y.data_ptr(),
        X.stride(0), Y.stride(0),
        N, D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return Y

# Example usage
if __name__ == "__main__":
    import torch.nn.functional as F
    import math

    # Create example input tensors
    X = torch.randn(128, 512, device='cuda', dtype=torch.float16)
    W = torch.randn(512, device='cuda', dtype=torch.float16)
    B = torch.randn(512, device='cuda', dtype=torch.float16)

    # Reference implementation
    ref_output = F.layer_norm(X, (X.shape[-1],), W, B, 1e-5).to(X.dtype)

    # Triton implementation
    triton_output = kernel(X, W, B)

    # Check if outputs are the same
    assert torch.allclose(ref_output, triton_output, atol=1e-3, rtol=1e-3)
    print("Outputs match within tolerance.")
