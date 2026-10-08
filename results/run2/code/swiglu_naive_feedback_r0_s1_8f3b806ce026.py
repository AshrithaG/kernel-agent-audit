import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel for SwiGLU
@triton.jit
def swiglu_kernel(
    X, Y,
    X.stride_0, X.stride_1, X.stride_2,
    Y.stride_0, Y.stride_1, Y.stride_2,
    N,
    BLOCK_SIZE: tl.constexpr
):
    # Get the position in the output tensor
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load the input values
    x_offset = offsets[:, None] * X.stride_0 + tl.arange(0, 32)[None, :]
    x = tl.load(X + x_offset, mask=mask[:, None], other=0.0)

    # Split the input into two halves
    n = N // 2
    x1 = x[:, :n]
    x2 = x[:, n:]

    # Apply SiLU activation to the first half
    x1_silu = x1 * tl.sigmoid(x1)

    # Multiply the SiLU result with the second half
    y = x1_silu * x2

    # Store the result in the output tensor
    y_offset = offsets[:, None] * Y.stride_0 + tl.arange(0, 32)[None, :]
    tl.store(Y + y_offset, y, mask=mask[:, None])

# Define the kernel wrapper function
def kernel(X):
    # Get the shape of the input tensor
    B, H, N = X.shape

    # Calculate the grid size
    grid_size = triton.cdiv(N, 32)

    # Define the output tensor
    Y = torch.empty_like(X)

    # Launch the Triton kernel
    swiglu_kernel[grid_size, 32](
        X.data_ptr(), Y.data_ptr(),
        X.stride(0), X.stride(1), X.stride(2),
        Y.stride(0), Y.stride(1), Y.stride(2),
        N,
        BLOCK_SIZE=32
    )

    return Y

# Example usage
if __name__ == "__main__":
    # Create a random input tensor of shape (B, H, N) with float16 dtype
    B, H, N = 8, 16, 64
    X = torch.rand((B, H, N), device='cuda', dtype=torch.float16)

    # Run the kernel
    Y = kernel(X)

    # Print the shapes and dtypes
    print("Input shape:", X.shape, "dtype:", X.dtype)
    print("Output shape:", Y.shape, "dtype:", Y.dtype)

    # Verify the correctness
    ref_Y = (torch.nn.functional.silu(X[..., :N//2].float()) * X[..., N//2:].float()).to(X.dtype)
    assert torch.allclose(Y, ref_Y, atol=1e-3, rtol=1e-3), "The output does not match the reference implementation"
    print("Verification passed!")
