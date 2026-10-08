import torch
import triton
import triton.language as tl
import math

# Triton kernel for rotary position embedding
@triton.jit
def rope_kernel(
    X, C, S,
    stride_xb, stride_xh, stride_xd,
    stride_cb, stride_cd,
    stride_sb, stride_sd,
    N, H, D
):
    # Thread identifiers
    BLOCK_SIZE = 128
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    block_id = pid * BLOCK_SIZE + bid
    if block_id >= N * H:
        return
    
    # Calculate indices
    b = block_id // H
    h = block_id % H
    d = tl.arange(0, D, block_size=BLOCK_SIZE)

    # Load data
    x1 = tl.load(X + b * stride_xb + h * stride_xh + d * stride_xd, mask=d < D // 2, other=0.0)
    x2 = tl.load(X + b * stride_xb + h * stride_xh + (d + D // 2) * stride_xd, mask=d < D // 2, other=0.0)
    c = tl.load(C + b * stride_cb + d * stride_cd, mask=d < D // 2, other=0.0)
    s = tl.load(S + b * stride_sb + d * stride_sd, mask=d < D // 2, other=0.0)

    # Compute rotated embeddings
    x1_rotated = x1 * c - x2 * s
    x2_rotated = x2 * c + x1 * s

    # Store results
    tl.store(X + b * stride_xb + h * stride_xh + d * stride_xd, x1_rotated, mask=d < D // 2)
    tl.store(X + b * stride_xb + h * stride_xh + (d + D // 2) * stride_xd, x2_rotated, mask=d < D // 2)

# Triton wrapper function
def kernel(X, C, S):
    N, H, D = X.shape
    assert D % 2 == 0, "Dimension must be even"
    assert C.shape == (N, D // 2) and S.shape == (N, D // 2), "Cos and Sin shapes must match"

    # Define grid and block dimensions
    grid = (triton.cdiv(N * H, 128),)
    block = (128,)

    # Launch kernel
    rope_kernel[grid, block](
        X, C, S,
        X.stride(0), X.stride(1), X.stride(2),
        C.stride(0), C.stride(1),
        S.stride(0), S.stride(1),
        N, H, D
    )
    return X

# Example usage
if __name__ == "__main__":
    # Create example input tensors
    tokens = 1024
    heads = 16
    dim = 128
    cos = torch.randn(tokens, dim // 2, device='cuda', dtype=torch.float16)
    sin = torch.randn(tokens, dim // 2, device='cuda', dtype=torch.float16)
    x = torch.randn(tokens, heads, dim, device='cuda', dtype=torch.float16)

    # Apply the kernel
    result = kernel(x.clone(), cos, sin)

    # Reference implementation
    reference_result = rope(x.clone(), cos, sin)

    # Check if results are the same
    assert torch.allclose(result, reference_result), "The results do not match"

    print("Kernel output matches reference implementation.")
