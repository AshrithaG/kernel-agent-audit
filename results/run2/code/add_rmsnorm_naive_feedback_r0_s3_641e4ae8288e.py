import torch
import triton
import triton.language as tl
import math

# Triton kernel for adding residual and applying RMSNorm
@triton.jit
def rmsnorm_kernel(
    X, R, W, Out,
    stride_xz, stride_xm, stride_xn,
    stride_rz, stride_rm, stride_rn,
    stride_wz, stride_wn,
    stride_oz, stride_om, stride_on,
    N, M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    bid_z = pid // (BLOCK_SIZE_M * BLOCK_SIZE_N)
    bid_m = (pid % (BLOCK_SIZE_M * BLOCK_SIZE_N)) // BLOCK_SIZE_N
    bid_n = (pid % (BLOCK_SIZE_M * BLOCK_SIZE_N)) % BLOCK_SIZE_N

    # Load data from global memory
    off_x = bid_z * stride_xz + bid_m * stride_xm + bid_n * stride_xn
    off_r = bid_z * stride_rz + bid_m * stride_rm + bid_n * stride_rn
    off_w = bid_z * stride_wz + bid_n * stride_wn
    off_o = bid_z * stride_oz + bid_m * stride_om + bid_n * stride_on

    X = tl.load(X + off_x, mask=(bid_m < M) & (bid_n < N), other=0.0)
    R = tl.load(R + off_r, mask=(bid_m < M) & (bid_n < N), other=0.0)
    W = tl.load(W + off_w, mask=(bid_n < N), other=0.0)

    # Add residual
    H = X + R

    # Compute mean of squares
    sum_squares = tl.sum(H**2, axis=1)
    mean_squares = sum_squares / N

    # Add epsilon and compute inverse square root
    inv_sqrt_mean_squares = 1.0 / tl.sqrt(mean_squares + 1e-6)

    # Apply RMSNorm
    Out = H * inv_sqrt_mean_squares * W

    # Store result back to global memory
    tl.store(Out + off_o, Out, mask=(bid_m < M) & (bid_n < N))

# Triton wrapper function
def kernel(X, R, W):
    # Ensure inputs are float16 and on the same device
    X = X.to(torch.float16)
    R = R.to(torch.float16)
    W = W.to(torch.float16)

    # Get dimensions
    B, M, N = X.shape

    # Define grid and block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 32
    grid_size = (B * (M // BLOCK_SIZE_M + (M % BLOCK_SIZE_M != 0)) *
                  (N // BLOCK_SIZE_N + (N % BLOCK_SIZE_N != 0)), )

    # Allocate output tensor
    Out = torch.empty_like(X)

    # Launch Triton kernel
    rmsnorm_kernel[grid_size](
        X, R, W, Out,
        stride_xz=X.stride(0), stride_xm=X.stride(1), stride_xn=X.stride(2),
        stride_rz=R.stride(0), stride_rm=R.stride(1), stride_rn=R.stride(2),
        stride_wz=W.stride(0), stride_wn=W.stride(1),
        stride_oz=Out.stride(0), stride_om=Out.stride(1), stride_on=Out.stride(2),
        N=N, M=M,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return Out

# Example usage
if __name__ == "__main__":
    # Create example inputs
    B, M, N = 2, 512, 768
    X = torch.randn(B, M, N, device='cuda', dtype=torch.float16)
    R = torch.randn(B, M, N, device='cuda', dtype=torch.float16)
    W = torch.randn(B, N, device='cuda', dtype=torch.float16)

    # Run Triton kernel
    out_triton = kernel(X, R, W)

    # Run reference implementation
    out_ref = (X.float() + R.float()) * torch.rsqrt((X.float() + R.float()).pow(2).mean(-1, keepdim=True) + 1e-6) * W.float()

    # Check if outputs are close
    assert torch.allclose(out_triton, out_ref, atol=1e-3, rtol=1e-3), "Outputs are not close"

    print("Triton kernel output matches reference implementation.")
