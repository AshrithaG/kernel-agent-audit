import torch
import triton
import triton.language as tl
import math

@triton.jit
def rmsnorm_kernel(
    X, W, Out,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_w, stride_out,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N, BLOCK_SIZE_M)
    num_pid_n = (X.shape[1] + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    block_id_m = pid // num_pid_n
    block_id_n = pid % num_pid_n

    rm = block_id_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = rm < N
    mask_n = rn < X.shape[1]

    x = tl.load(X + rm[:, None] * stride_xm + rn[None, :] * stride_xn, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    w = tl.load(W + rn * stride_w, mask=mask_n, other=0.0)

    mean = tl.sum(x ** 2, axis=1) / X.shape[1]
    rmean = tl.sqrt(mean + 1e-6)

    out = x / rmean[:, None] * w[None, :]
    tl.store(Out + rm[:, None] * stride_ym + rn[None, :] * stride_yn, out, mask=mask_m[:, None] & mask_n[None, :])

def kernel(X, W):
    N = X.shape[0]
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128

    grid = (triton.cdiv(N, BLOCK_SIZE_M) * ((X.shape[1] + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N), )
    Out = torch.empty_like(X)
    rmsnorm_kernel[grid](
        X, W, Out,
        X.stride(0), X.stride(1),
        Out.stride(0), Out.stride(1),
        W.stride(0), Out.stride(0),
        N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return Out

# Example usage:
# X = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
# W = torch.randn(512, device='cuda', dtype=torch.float16)
# result = kernel(X, W)
