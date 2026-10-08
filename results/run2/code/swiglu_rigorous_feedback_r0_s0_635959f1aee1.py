import torch
import triton
import triton.language as tl

@triton.jit
def swiglu_kernel(x_ptr, y_ptr, N, M, BLOCK_SIZE: tl.constexpr):
    # Calculate the index of the current thread
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl threadIdx.x
    # Determine if the index is within bounds
    mask = idx < N * M
    # Load the first half of the input
    x1 = tl.load(x_ptr + idx, mask=mask, other=0.0)
    # Load the second half of the input
    x2 = tl.load(x_ptr + idx + N * M, mask=mask, other=0.0)
    # Apply the SiLU activation function to the first half
    x1 = x1 * tl.sigmoid(x1)
    # Multiply the first half by the second half
    y = x1 * x2
    # Store the result
    tl.store(y_ptr + idx, y, mask=mask)

def kernel(x):
    # Get the shape of the input tensor
    N, M = x.shape[:-1], x.shape[-1] // 2
    # Flatten the input tensor except for the last dimension
    x_flat = x.view(-1, M * 2)
    # Allocate memory for the output tensor
    y_flat = torch.empty_like(x_flat)
    # Define the block size
    BLOCK_SIZE = 1024
    # Launch the Triton kernel
    grid = lambda opt: (triton.cdiv(x_flat.numel(), BLOCK_SIZE),)
    swiglu_kernel[grid](x_flat, y_flat, N.numel(), M, BLOCK_SIZE=BLOCK_SIZE)
    # Reshape the output tensor to match the original input shape
    y = y_flat.view(*N, M * 2)
    return y

# Example usage
if __name__ == "__main__":
    import torch.nn.functional as F
    import math

    # Create a random input tensor of shape (B, H, L, 2 * D)
    B, H, L, D = 8, 16, 32, 128
    x = torch.randn(B, H, L, 2 * D, device='cuda', dtype=torch.float16)

    # Reference implementation
    def reference(x):
        n = x.shape[-1] // 2
        return (F.silu(x[..., :n].float()) * x[..., n:].float()).to(x.dtype)

    # Triton implementation
    y_triton = kernel(x)
    y_ref = reference(x)

    # Check if the outputs are equal within floating-point tolerance
    assert torch.allclose(y_triton, y_ref, atol=1e-3, rtol=1e-3), "Outputs do not match"

    print("Triton kernel implementation is correct.")
