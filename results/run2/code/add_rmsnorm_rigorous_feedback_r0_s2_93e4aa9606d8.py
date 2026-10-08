import torch
import triton
import triton.language as tl
import math

@triton.jit
def rmsnorm_kernel(
    X, R, W, Out,
    stride_xz, stride_xm, stride_xn,
    stride_rz, stride_rm, stride_rn,
    stride_wz, stride_wn,
    stride_oz, stride_om, stride_on,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    EPS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid_z = tl.program_id(axis=1)
    bid_m = tl.program_id(axis=2)

    # Calculate the row and column indices
    row = bid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col = bid_z * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Load data from global memory
    X = X + R
    x = tl.load(X + pid * stride_xz + row * stride_xm + col * stride_xn, mask=row < N, other=0.0)
    w = tl.load(W + pid * stride_wz + col * stride_wn, mask=col < N, other=0.0)

    # Compute the mean of squares
    mean_squares = tl.sum(x * x, axis=1)
    mean_squares = tl.where(mean_squares == 0.0, 1e-6, mean_squares)
    rstd = tl.rsqrt(mean_squares + EPS)

    # Apply RMSNorm
    x = x * rstd[:, None]
    x = x * w[None, :]

    # Store the result back to global memory
    tl.store(Out + pid * stride_oz + row * stride_om + col * stride_on, x, mask=row < N)

def kernel(X, R, W, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, EPS=1e-6):
    # Ensure inputs are on the same device
    assert X.device == R.device == W.device
    assert X.dtype == R.dtype == W.dtype == torch.float16

    # Get the dimensions
    B, M, N = X.shape

    # Create output tensor
    Out = torch.empty_like(X)

    # Define grid and block sizes
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
        B
    )
    blocks = (BLOCK_SIZE_M, BLOCK_SIZE_N)

    # Launch the kernel
    rmsnorm_kernel[grid](
        X, R, W, Out,
        X.stride(0), X.stride(1), X.stride(2),
        R.stride(0), R.stride(1), R.stride(2),
        W.stride(0), W.stride(1),
        Out.stride(0), Out.stride(1), Out.stride(2),
        N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        EPS=EPS,
    )

    return Out

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(32, 1024, 512, device='cuda', dtype=torch.float16)
    R = torch.randn(32, 1024, 512, device='cuda', dtype=torch.float16)
    W = torch.randn(32, 512, device='cuda', dtype=torch.float16)

    out = kernel(X, R, W)
    print(out.shape, out.dtype)
