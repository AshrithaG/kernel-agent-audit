import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel for SwiGLU
@triton.jit
def swiglu_kernel(
    X, Y,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    
    # Calculate the row and column indices
    row = pid // BLOCK_SIZE
    col = pid % BLOCK_SIZE
    
    # Load the input values
    offset = row * stride_xm + col * stride_xn
    x1 = tl.load(X + offset, mask=col < N // 2, other=0.0)
    x2 = tl.load(X + offset + N // 2 * stride_xn, mask=col < N // 2, other=0.0)
    
    # Apply SiLU activation to the first half
    x1_silu = x1 * tl.sigmoid(x1)
    
    # Multiply the SiLU result with the second half
    y = x1_silu * x2
    
    # Store the output values
    tl.store(Y + offset, y, mask=col < N // 2)

# Define the kernel wrapper function
def kernel(X, Y):
    # Get the shape of the input tensor
    M, N = X.shape
    
    # Define the grid and block sizes
    BLOCK_SIZE = 128
    grid = lambda opt: (triton.cdiv(M, BLOCK_SIZE),)
    
    # Launch the Triton kernel
    swiglu_kernel[grid](
        X, Y,
        X.stride(0), X.stride(1),
        Y.stride(0), Y.stride(1),
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )

# Reference implementation for testing
def reference(X):
    n = X.shape[-1] // 2
    return (torch.nn.functional.silu(X[..., :n].float()) * X[..., n:].float()).to(X.dtype)

# Example usage
if __name__ == "__main__":
    # Create a random input tensor
    X = torch.randn(1024, 256, device='cuda', dtype=torch.float16)
    Y = torch.empty_like(X)
    
    # Run the Triton kernel
    kernel(X, Y)
    
    # Run the reference implementation
    Z = reference(X)
    
    # Check if the outputs are equal within floating-point tolerance
    assert torch.allclose(Y, Z, atol=1e-3, rtol=1e-3)
    print("Test passed!")
