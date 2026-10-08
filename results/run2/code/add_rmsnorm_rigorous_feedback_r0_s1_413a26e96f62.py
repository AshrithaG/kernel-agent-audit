import torch
import triton
import triton.language as tl

# Triton kernel for adding residual and performing RMSNorm
@triton.jit
def rmsnorm_kernel(
    X, R, W, Out,
    stride_xz, stride_xm,
    stride_rz, stride_rm,
    stride_wz, stride_wn,
    stride_oz, stride_on,
    N, M,
    BLOCK_SIZE_Z: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    EPS: tl.constexpr
):
    # Define the program ID
    off_z = tl.program_id(axis=0)
    off_m = tl.program_id(axis=1)

    # Load the data into registers
    off_x = off_z * stride_xz + off_m * stride_xm
    off_r = off_z * stride_rz + off_m * stride_rm
    off_w = off_z * stride_wz + off_m * stride_wn
    off_o = off_z * stride_oz + off_m * stride_on

    x = tl.load(X + off_x, mask=off_m < M, other=0.0)
    r = tl.load(R + off_r, mask=off_m < M, other=0.0)
    w = tl.load(W + off_w, mask=off_m < M, other=0.0)

    # Compute the sum
    h = x + r

    # Compute the mean of squares
    sum_squares = tl.sum(h**2, axis=0)

    # Compute the RMSNorm
    rms = tl.sqrt(sum_squares / M + EPS)

    # Normalize and scale
    out = h / rms * w

    # Store the result
    tl.store(Out + off_o, out, mask=off_m < M)

# Triton kernel wrapper function
def kernel(X, R, W):
    assert X.shape == R.shape == W.shape, "Input shapes must match"
    assert X.ndim == 2, "Inputs must be 2D tensors"
    
    # Get the dimensions
    N, M = X.shape
    
    # Define the grid
    grid = (triton.cdiv(N, 32), triton.cdiv(M, 32))
    
    # Define the EPS value
    EPS = 1e-6
    
    # Launch the kernel
    rmsnorm_kernel[grid](
        X, R, W, X,
        X.stride(0), X.stride(1),
        R.stride(0), R.stride(1),
        W.stride(0), W.stride(1),
        X.stride(0), X.stride(1),
        N, M,
        BLOCK_SIZE_Z=32,
        BLOCK_SIZE_M=32,
        EPS=EPS,
    )
    
    return X

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
    R = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
    W = torch.randn(1024, 512, dtype=torch.float16, device='cuda')

    # Reference implementation
    ref_output = (X.float() + R.float()) * torch.rsqrt((X.float() + R.float()).pow(2).mean(-1, keepdim=True) + 1e-6) * W.float()
    ref_output = ref_output.to(X.dtype)

    # Triton implementation
    triton_output = kernel(X.clone(), R.clone(), W.clone())

    # Check if outputs are close
    print(torch.allclose(ref_output, triton_output))
