import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    X, W, Y,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N, BLOCK_M)
    num_pid_n = tl.cdiv(X.shape[1], BLOCK_N)
    
    # Compute row and column indices
    rm = pid // num_pid_n
    rn = pid % num_pid_n
    
    # Load data from global memory
    x = tl.load(X + rm * stride_xm + rn * stride_xn, mask=rn < X.shape[1], other=0.0)
    w = tl.load(W + rn * stride_yn, mask=rn < X.shape[1], other=0.0)
    
    # Compute mean of squares
    sum_squares = tl.sum(x * x, axis=1)
    mean_squares = sum_squares / N
    
    # Add epsilon and compute reciprocal square root
    rsum_squares = 1.0 / tl.sqrt(mean_squares + 1e-6)
    
    # Normalize and scale
    y = x * rsum_squares * w
    
    # Store result to global memory
    tl.store(Y + rm * stride_ym + rn * stride_yn, y, mask=rn < X.shape[1])

def kernel(X, W):
    assert X.is_cuda and W.is_cuda, "Inputs must be CUDA tensors"
    assert X.dtype == torch.float16 and W.dtype == torch.float16, "Inputs must be float16 tensors"
    
    N = X.shape[0]
    BLOCK_M = 32
    BLOCK_N = 128
    
    grid = lambda META: (
        triton.cdiv(X.shape[0], META['BLOCK_M']) *
        triton.cdiv(X.shape[1], META['BLOCK_N'])
    )
    
    Y = torch.empty_like(X)
    
    rmsnorm_kernel[grid](
        X, W, Y,
        X.stride(0), X.stride(1),
        Y.stride(0), Y.stride(1),
        N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )
    
    return Y

# Example usage
if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    X = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
    W = torch.randn(512, device='cuda', dtype=torch.float16)
    
    Y_ref = (X.float() * torch.rsqrt(X.float().pow(2).mean(-1, keepdim=True) + 1e-6) * W.float()).to(X.dtype)
    Y_triton = kernel(X, W)
    
    print(torch.allclose(Y_ref, Y_triton))
