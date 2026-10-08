import torch
import triton
import triton.language as tl

@triton.jit
def add_rmsnorm_kernel(
    X, R, W, Out, 
    stride_xz, stride_xm, stride_xn,
    stride_rz, stride_rm, stride_rn,
    stride_wz, stride_wn,
    stride_oz, stride_om, stride_on,
    N, M, Z,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr
):
    pid_z = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    # Load data from global memory to shared memory
    offset_x = pid_z * stride_xz + pid_m * stride_xm + pid_n * stride_xn
    offset_r = pid_z * stride_rz + pid_m * stride_rm + pid_n * stride_rn
    offset_w = pid_z * stride_wz + pid_n * stride_wn
    offset_o = pid_z * stride_oz + pid_m * stride_om + pid_n * stride_on

    x = tl.load(X + offset_x, mask=(pid_n < N), other=0.0)
    r = tl.load(R + offset_r, mask=(pid_n < N), other=0.0)
    w = tl.load(W + offset_w, mask=(pid_n < M), other=0.0)

    # Compute the sum
    h = x + r

    # Compute the mean of squares
    sum_squares = tl.sum(h * h, axis=0)

    # Compute the RMSNorm
    rmsnorm = tl.sqrt(sum_squares / N + 1e-6)

    # Apply the weight
    out = h / rmsnorm * w

    # Store the result back to global memory
    tl.store(Out + offset_o, out, mask=(pid_n < N))

# Triton kernel wrapper
def kernel(X, R, W):
    assert X.is_cuda and R.is_cuda and W.is_cuda, "Inputs must be CUDA tensors"
    assert X.dtype == torch.float16 and R.dtype == torch.float16 and W.dtype == torch.float16, "Inputs must be float16"
    assert X.shape == R.shape and X.shape[:-1] == W.shape, "Shape mismatch"

    Z, M, N = X.shape
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_M = 32

    grid = (
        triton.cdiv(N, BLOCK_SIZE_N),
        triton.cdiv(M, BLOCK_SIZE_M),
        Z
    )

    Out = torch.empty_like(X)

    add_rmsnorm_kernel[grid](
        X.data_ptr(), R.data_ptr(), W.data_ptr(), Out.data_ptr(),
        X.stride(0), X.stride(1), X.stride(2),
        R.stride(0), R.stride(1), R.stride(2),
        W.stride(0), W.stride(1),
        Out.stride(0), Out.stride(1), Out.stride(2),
        N, M, Z,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M
    )

    return Out

# Example usage
if __name__ == "__main__":
    X = torch.randn(4, 32, 128, device='cuda', dtype=torch.float16)
    R = torch.randn(4, 32, 128, device='cuda', dtype=torch.float16)
    W = torch.randn(32, 128, device='cuda', dtype=torch.float16)

    out_triton = kernel(X, R, W)
    out_ref = (X.float() + R.float()) * torch.rsqrt((X.float() + R.float()).pow(2).mean(-1, keepdim=True) + 1e-6) * W.float()

    print("Triton output:", out_triton)
    print("Reference output:", out_ref)
    print("Max difference:", torch.max(torch.abs(out_triton - out_ref)))
