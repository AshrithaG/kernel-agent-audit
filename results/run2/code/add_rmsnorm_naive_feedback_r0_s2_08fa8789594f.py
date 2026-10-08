import torch
import triton
import triton.language as tl
import math

@triton.jit
def rmsnorm_kernel(
    X, R, W, Out,
    stride_xz, stride_xm,
    stride_rz, stride_rm,
    stride_wz, stride_wn,
    stride_oz, stride_on,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    rows = tl.arange(0, BLOCK_SIZE)
    mask_z = rows < N

    # Load x, r, w
    x = tl.load(X + rows[:, None] * stride_xz + block_start * stride_xm, mask=mask_z[:, None], other=0.0)
    r = tl.load(R + rows[:, None] * stride_rz + block_start * stride_rm, mask=mask_z[:, None], other=0.0)
    w = tl.load(W + rows[:, None] * stride_wz, mask=mask_z[:, None], other=0.0)

    # Add residual
    h = x + r

    # Compute mean of squares
    sum_squares = tl.sum(h * h, axis=1)
    mean_squares = sum_squares / N

    # Compute RMSNorm
    inv_sqrt_mean_squares = 1.0 / tl.sqrt(mean_squares + 1e-6)
    normed_h = h * inv_sqrt_mean_squares[:, None]

    # Scale by weight
    scaled_h = normed_h * w

    # Store result
    tl.store(Out + rows[:, None] * stride_oz + block_start * stride_on, scaled_h, mask=mask_z[:, None])

# Triton wrapper function
def kernel(X, R, W):
    N = X.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)

    X = X.contiguous().half()
    R = R.contiguous().half()
    W = W.contiguous().half()
    Out = torch.empty_like(X, dtype=torch.half)

    rmsnorm_kernel[grid](
        X.data_ptr(), R.data_ptr(), W.data_ptr(), Out.data_ptr(),
        X.stride(0), X.stride(1),
        R.stride(0), R.stride(1),
        W.stride(0), W.stride(1),
        Out.stride(0), Out.stride(1),
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return Out.to(X.dtype)

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(32, 128, device='cuda', dtype=torch.float16)
    r = torch.randn(32, 128, device='cuda', dtype=torch.float16)
    w = torch.randn(128, device='cuda', dtype=torch.float16)

    out_triton = kernel(x, r, w)
    out_ref = (x + r).float() * torch.rsqrt((x + r).pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()

    print(torch.allclose(out_triton, out_ref))
