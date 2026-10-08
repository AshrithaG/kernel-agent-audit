import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel for SwiGLU
@triton.jit
def swiglu_kernel(
    X, Y,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the index of the current thread
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl threadIdx.x
    
    # Load the first half and second half of the input
    first_half = tl.load(X + idx, mask=(idx < N), other=0.0)
    second_half = tl.load(X + idx + N, mask=(idx + N < N * 2), other=0.0)
    
    # Apply SiLU to the first half
    first_half_silu = first_half * tl.sigmoid(first_half)
    
    # Multiply the SiLU result with the second half
    result = first_half_silu * second_half
    
    # Store the result in the output tensor
    tl.store(Y + idx, result, mask=(idx < N))

# Define the wrapper function to handle the kernel launch
def kernel(X):
    # Ensure the input is a float16 tensor
    assert X.dtype == torch.float16
    
    # Get the shape of the input tensor
    B, T, C = X.shape
    
    # Calculate the number of elements in the last dimension
    N = C // 2
    
    # Create an output tensor of the same shape as the input
    Y = torch.empty_like(X)
    
    # Define the grid and block sizes for the kernel launch
    BLOCK_SIZE = 128
    grid_size = (B * T) * triton.cdiv(N, BLOCK_SIZE)
    
    # Launch the kernel
    swiglu_kernel[grid_size, BLOCK_SIZE](
        X, Y,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return Y

# Reference implementation for verification
def reference(X):
    n = X.shape[-1] // 2
    return (torch.nn.functional.silu(X[..., :n].float()) * X[..., n:].float()).to(X.dtype)

# Example usage
if __name__ == "__main__":
    # Create a random input tensor of shape (B, T, C) with float16 dtype
    B, T, C = 32, 1024, 512
    X = torch.rand(B, T, C, device="cuda", dtype=torch.float16)
    
    # Run the Triton kernel and the reference implementation
    Y_triton = kernel(X)
    Y_ref = reference(X)
    
    # Verify that the results are the same within floating-point tolerance
    assert torch.allclose(Y_triton, Y_ref, atol=1e-3, rtol=1e-3)
    
    print("Triton kernel output matches reference implementation.")
