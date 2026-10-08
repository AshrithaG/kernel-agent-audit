import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel(
    X, C, S, 
    stride_xz, stride_xh, stride_xd,
    stride_cz, stride_cd,
    stride_sz, stride_sd,
    Z, H, D
):
    # Get the position and head indices
    z = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D // 2)

    # Load the input data
    x1 = tl.load(X + z * stride_xz + h * stride_xh + d * stride_xd, mask=z < Z, other=0.0)
    x2 = tl.load(X + z * stride_xz + h * stride_xh + (d + D // 2) * stride_xd, mask=z < Z, other=0.0)
    
    # Load the cos and sin values
    c = tl.load(C + z * stride_cz + d * stride_cd, mask=z < Z, other=0.0)
    s = tl.load(S + z * stride_sz + d * stride_sd, mask=z < Z, other=0.0)

    # Apply the rotation
    x1_rotated = x1 * c - x2 * s
    x2_rotated = x2 * c + x1 * s

    # Store the results back to the output buffer
    tl.store(X + z * stride_xz + h * stride_xh + d * stride_xd, x1_rotated, mask=z < Z)
    tl.store(X + z * stride_xz + h * stride_xh + (d + D // 2) * stride_xd, x2_rotated, mask=z < Z)

def kernel(X, C, S):
    # X: [tokens, heads, dim]
    # C, S: [tokens, dim // 2]
    Z, H, D = X.shape
    BLOCK_Z = 128
    BLOCK_H = 32

    grid = (triton.cdiv(Z, BLOCK_Z), triton.cdiv(H, BLOCK_H))
    rope_kernel[grid](
        X, C, S,
        X.stride(0), X.stride(1), X.stride(2),
        C.stride(0), C.stride(1),
        S.stride(0), S.stride(1),
        Z, H, D
    )
    return X

# Example usage:
if __name__ == "__main__":
    import torch
    tokens, heads, dim = 1024, 8, 64
    X = torch.rand((tokens, heads, dim), device='cuda', dtype=torch.float16)
    C = torch.cos(torch.linspace(0, 10, tokens, device='cuda', dtype=torch.float16)).unsqueeze(-1).expand(-1, dim // 2)
    S = torch.sin(torch.linspace(0, 10, tokens, device='cuda', dtype=torch.float16)).unsqueeze(-1).expand(-1, dim // 2)

    result = kernel(X.clone(), C.clone(), S.clone())
    expected = torch.cat([X[..., :dim//2] * C - X[..., dim//2:] * S, X[..., dim//2:] * C + X[..., :dim//2] * S], dim=-1)
    assert torch.allclose(result, expected, atol=1e-3, rtol=1e-3)
    print("Test passed!")
