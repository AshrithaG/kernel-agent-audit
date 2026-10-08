import torch
import triton
import triton.language as tl
import math

@triton.jit
def add_rmsnorm_kernel(
    X, R, W, Out,
    stride_xz, stride_xm,
    stride_rz, stride_rm,
    stride_wz, stride_wn,
    stride_oz, stride_on,
    N: tl.constexpr,
    M: tl.constexpr,
    BLOCK_SIZE_Z: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Indices
    off_z = tl.program_id(axis=0)
    off_m = tl.program_id(axis=1)

    # Load X, R, W
    Xs = tl.load(X + off_z * stride_xz + off_m * stride_xm, mask=off_z < N & off_m < M, other=0.0)
    Rs = tl.load(R + off_z * stride_rz + off_m * stride_rm, mask=off_z < N & off_m < M, other=0.0)
    Ws = tl.load(W + off_z * stride_wz, mask=off_z < N, other=0.0)

    # Add residual
    Hs = Xs + Rs

    # Compute mean of squares
    mean_square = tl.sum(Hs**2, axis=1)
    mean_square = tl.sum(mean_square, axis=0) / (N * M)
    mean_square = tl.load(mean_square.to_ptr(), mask=tl.arange(0, 1) < 1, other=0.0)

    # RMSNorm
    rms = tl.sqrt(mean_square + 1e-6)
    Hs = Hs / rms

    # Apply weight
    Hs = Hs * Ws

    # Store result
    tl.store(Out + off_z * stride_oz + off_m * stride_on, Hs, mask=off_z < N & off_m < M)

# Define the kernel function
def kernel(X, R, W):
    N, M = X.shape
    assert R.shape == (N, M)
    assert W.shape == (N,)
    assert X.dtype == torch.float16
    assert R.dtype == torch.float16
    assert W.dtype == torch.float16

    # Create output tensor
    Out = torch.empty_like(X)

    # Define grid
    BLOCK_SIZE_Z = 128
    BLOCK_SIZE_M = 32
    grid_z = triton.cdiv(N, BLOCK_SIZE_Z)
    grid_m = triton.cdiv(M, BLOCK_SIZE_M)
    grid = (grid_z, grid_m)

    # Launch kernel
    add_rmsnorm_kernel[grid](
        X, R, W, Out,
        X.stride(0), X.stride(1),
        R.stride(0), R.stride(1),
        W.stride(0), W.stride(1),
        Out.stride(0), Out.stride(1),
        N=N, M=M,
        BLOCK_SIZE_Z=BLOCK_SIZE_Z,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )

    return Out

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    R = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    W = torch.randn(1024, device='cuda', dtype=torch.float16)

    out_ref = add_rmsnorm(X, R, W)
    out_triton = kernel(X, R, W)

    print(torch.allclose(out_ref, out_triton, atol=1e-3))
