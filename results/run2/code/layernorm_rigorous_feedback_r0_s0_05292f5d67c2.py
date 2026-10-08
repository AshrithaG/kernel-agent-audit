import torch
import triton
import triton.language as tl

# Define the Triton kernel for layer normalization
@triton.jit
def layer_norm_kernel(
    X, W, B, Y,
    stride_xz, stride_yz,
    N, C,
    BLOCK_SIZE: tl.constexpr
):
    # Each thread handles one element of the output
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < C

    # Load data from global memory
    x = tl.load(X + bid * stride_xz + idx, mask=mask, other=0.0)
    w = tl.load(W + idx, mask=mask, other=0.0)
    b = tl.load(B + idx, mask=mask, other=0.0)

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / N
    var = tl.sum((x - mean)**2, axis=0) / N + 1e-5

    # Normalize and scale
    inv_var = tl.rsqrt(var)
    normalized = (x - mean) * inv_var
    result = normalized * w + b

    # Store result to global memory
    tl.store(Y + bid * stride_yz + idx, result, mask=mask)

# Define the wrapper function to launch the Triton kernel
def kernel(X, W, B):
    N, C = X.shape
    BLOCK_SIZE = 128

    # Ensure inputs are on the same device
    X = X.contiguous()
    W = W.contiguous()
    B = B.contiguous()

    # Allocate output tensor
    Y = torch.empty_like(X)

    # Launch the Triton kernel
    grid = (triton.cdiv(C, BLOCK_SIZE), N)
    layer_norm_kernel[grid](
        X, W, B, Y,
        X.stride(0), Y.stride(0),
        N, C,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return Y

# Example usage
if __name__ == "__main__":
    import torch.nn.functional as F

    # Create example inputs
    X = torch.randn(32, 1024, dtype=torch.float16, device='cuda')
    W = torch.randn(1024, dtype=torch.float16, device='cuda')
    B = torch.randn(1024, dtype=torch.float16, device='cuda')

    # Reference implementation
    reference_output = F.layer_norm(X.float(), (X.shape[-1],), W.float(), B.float(), 1e-5).to(X.dtype)

    # Triton implementation
    triton_output = kernel(X, W, B)

    # Check if outputs are close
    assert torch.allclose(reference_output, triton_output, atol=1e-3, rtol=1e-3)
    print("Outputs match within tolerance!")
