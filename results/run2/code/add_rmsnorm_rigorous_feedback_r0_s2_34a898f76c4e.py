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
    w = tl.load(W + bid_z * stride_wz + col * stride_wn, mask=col < N, other=0.0)

    # Compute the sum of squares
    sum_squares = tl.sum(x * x, axis=1)

    # Compute the RMS norm
    rms = tl.sqrt(sum_squares + EPS)

    # Normalize and scale
    x_normalized = x / rms[:, None]
    out = x_normalized * w

    # Store the result back to global memory
    tl.store(Out + pid * stride_oz + row * stride_om + col * stride_on, out, mask=row < N)

def kernel(X, R, W, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, EPS=1e-6):
    assert X.is_cuda and R.is_cuda and W.is_cuda, "All inputs must be CUDA tensors"
    assert X.dtype == torch.float16 and R.dtype == torch.float16 and W.dtype == torch.float16, "All inputs must be of type float16"
    assert X.ndim == 3 and R.ndim == 3 and W.ndim == 2, "Invalid input shapes"

    B, M, N = X.shape
    Z, _ = W.shape

    # Ensure that the output shape matches the expected shape
    Out = torch.empty_like(X)

    grid = (
        triton.cdiv(B, BLOCK_SIZE_M),
        Z,
        triton.cdiv(M, BLOCK_SIZE_N),
    )

    rmsnorm_kernel[grid](
        X, R, W, Out,
        stride_xz=X.stride(0), stride_xm=X.stride(1), stride_xn=X.stride(2),
        stride_rz=R.stride(0), stride_rm=R.stride(1), stride_rn=R.stride(2),
        stride_wz=W.stride(0), stride_wn=W.stride(1),
        stride_oz=Out.stride(0), stride_om=Out.stride(1), stride_on=Out.stride(2),
        N=N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        EPS=EPS,
    )

    return Out

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(32, 512, 768, device='cuda', dtype=torch.float16)
    R = torch.randn(32, 512, 768, device='cuda', dtype=torch.float16)
    W = torch.randn(768, device='cuda', dtype=torch.float16)

    result = kernel(X, R, W)
    print(result.shape, result.dtype)
