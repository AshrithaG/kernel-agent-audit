import torch
import triton
import triton.language as tl
import math

# Triton kernel for LayerNorm
@triton.jit
def layer_norm_kernel(
    X, W, B, Y,
    N, D,
    BLOCK_SIZE: tl.constexpr
):
    # Each thread handles a single element in the output
    idx = tl.program_id(0)
    row = idx // D
    col = idx % D

    # Load the input value
    x = tl.load(X + row * D + col, mask=(col < D), other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / D

    # Compute variance
    var = tl.sum((x - mean) ** 2, axis=0) / D

    # Compute normalized value
    inv_var = tl.rsqrt(var + 1e-5)
    normalized_x = (x - mean) * inv_var

    # Apply weight and bias
    y = normalized_x * tl.load(W + col, mask=(col < D), other=0.0) + tl.load(B + col, mask=(col < D), other=0.0)

    # Store the result
    tl.store(Y + row * D + col, y, mask=(col < D))

# Function to launch the Triton kernel
def kernel(X, W, B):
    N, D = X.shape
    BLOCK_SIZE = 1024

    # Allocate output tensor
    Y = torch.empty_like(X)

    # Launch the kernel
    grid = lambda META: (tl.cdiv(N * D, META['BLOCK_SIZE']),)
    layer_norm_kernel[grid](X, W, B, Y, N, D, BLOCK_SIZE=BLOCK_SIZE)

    return Y

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    W = torch.randn(1024, device='cuda', dtype=torch.float16)
    B = torch.randn(1024, device='cuda', dtype=torch.float16)

    Y_triton = kernel(X, W, B)
    Y_ref = F.layer_norm(X.float(), (X.shape[-1],), W.float(), B.float(), 1e-5).to(X.dtype)

    print("Triton output:", Y_triton)
    print("Reference output:", Y_ref)

    # Check if outputs are close
    assert torch.allclose(Y_triton, Y_ref, atol=1e-3, rtol=1e-3), "Outputs do not match"
