import torch
import triton
import triton.language as tl
import math

# Triton kernel for adding residual and performing RMSNorm
@triton.jit
def rmsnorm_kernel(
    X, R, W, Out,
    H, stride_xz, stride_xm, stride_rz, stride_rm, stride_wz, stride_oz, stride_om,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Get the thread ID
    off_m = tl.program_id(axis=0) * BLOCK_SIZE_M + tl.thread_idx(axis=0)
    off_n = tl.program_id(axis=1) * BLOCK_SIZE_N + tl.thread_idx(axis=1)

    # Load the data
    x = tl.load(X + off_m * stride_xm + off_n * stride_xz, mask=off_m < H, other=0.0)
    r = tl.load(R + off_m * stride_rm + off_n * stride_rz, mask=off_m < H, other=0.0)
    w = tl.load(W + off_n * stride_wz, mask=off_n < BLOCK_SIZE_N, other=0.0)

    # Add residual
    h = x + r

    # Compute the mean of squares
    sum_squares = tl.sum(h * h, axis=1)
    mean_squares = sum_squares / BLOCK_SIZE_N

    # Compute the RMSNorm
    inv_sqrt_mean_squares = 1.0 / tl.sqrt(mean_squares + 1e-6)
    normalized_h = h * inv_sqrt_mean_squares

    # Scale by weight
    scaled_h = normalized_h * w

    # Store the result
    tl.store(Out + off_m * stride_om + off_n * stride_oz, scaled_h, mask=off_m < H)

# Wrapper function to call the Triton kernel
def kernel(X, R, W):
    assert X.is_cuda and R.is_cuda and W.is_cuda, "Inputs must be CUDA tensors"
    assert X.dtype == torch.float16 and R.dtype == torch.float16 and W.dtype == torch.float16, "Inputs must be float16 tensors"
    
    B, H, N = X.shape
    Out = torch.empty_like(X)

    # Define the grid and block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 32
    grid = (triton.cdiv(H, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch the Triton kernel
    rmsnorm_kernel[grid](
        X, R, W, Out,
        H,
        X.stride(0), X.stride(1), X.stride(2),
        R.stride(0), R.stride(1), R.stride(2),
        W.stride(0),
        Out.stride(0), Out.stride(1), Out.stride(2),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return Out

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(2, 512, 768, device='cuda', dtype=torch.float16)
    R = torch.randn(2, 512, 768, device='cuda', dtype=torch.float16)
    W = torch.randn(768, device='cuda', dtype=torch.float16)

    out = kernel(X, R, W)
    print(out.shape, out.dtype)
