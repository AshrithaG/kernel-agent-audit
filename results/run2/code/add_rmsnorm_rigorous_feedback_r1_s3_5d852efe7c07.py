import torch
import triton
import triton.language as tl
import math

@triton.jit
def add_rmsnorm_kernel(
    X, R, W, Out,
    stride_xz, stride_xm,
    stride_rz, stride_rm,
    stride_wz,
    stride_oz, stride_on,
    N: tl.constexpr,
    M: tl.constexpr,
    BLOCK_SIZE_Z: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Indices
    off_z = tl.program_id(axis=0) * BLOCK_SIZE_Z + tl.arange(0, BLOCK_SIZE_Z)
    off_m = tl.program_id(axis=1) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # Load X, R, W
    Xs = tl.load(X + off_z[:, None] * stride_xz + off_m[None, :] * stride_xm, mask=(off_z[:, None] < N) & (off_m[None, :] < M), other=0.0)
    Rs = tl.load(R + off_z[:, None] * stride_rz + off_m[None, :] * stride_rm, mask=(off_z[:, None] < N) & (off_m[None, :] < M), other=0.0)
    Ws = tl.load(W + off_z * stride_wz, mask=off_z < N, other=0.0)

    # Add residual
    Hs = Xs + Rs

    # Compute mean of squares
    sum_squares = tl.sum(Hs**2, axis=1)
    mean_squares = sum_squares / M

    # Compute RMSNorm
    inv_sqrt_mean_squares = 1.0 / tl.sqrt(mean_squares + 1e-6)
    NormHs = Hs * inv_sqrt_mean_squares[:, None]

    # Scale by W
    Outs = NormHs * Ws[:, None]

    # Store output
    tl.store(Out + off_z[:, None] * stride_oz + off_m[None, :] * stride_on, Outs, mask=(off_z[:, None] < N) & (off_m[None, :] < M))

def kernel(X, R, W):
    N, M = X.shape
    BLOCK_SIZE_Z = 128
    BLOCK_SIZE_M = 32

    # Define grid
    grid = (
        triton.cdiv(N, BLOCK_SIZE_Z),
        triton.cdiv(M, BLOCK_SIZE_M),
    )

    # Launch kernel
    add_rmsnorm_kernel[grid](
        X, R, W, X,  # Reuse X as output buffer
        X.stride(0), X.stride(1),
        R.stride(0), R.stride(1),
        W.stride(0),
        X.stride(0), X.stride(1),
        N=N, M=M,
        BLOCK_SIZE_Z=BLOCK_SIZE_Z,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )

    return X

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
    R = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
    W = torch.randn(1024, dtype=torch.float16, device='cuda')

    # Reference implementation
    ref_output = (X.float() + R.float()).rsqrt((X.float() + R.float()).pow(2).mean(-1, keepdim=True) + 1e-6) * W.float()
    ref_output = ref_output.to(X.dtype)

    # Triton implementation
    triton_output = kernel(X.clone(), R.clone(), W.clone())

    # Check correctness
    assert torch.allclose(ref_output, triton_output, atol=1e-3, rtol=1e-3)

    print("Reference output:", ref_output)
    print("Triton output:", triton_output)
