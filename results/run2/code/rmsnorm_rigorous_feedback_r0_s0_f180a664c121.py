import torch
import triton
import triton.language as tl
import math

@triton.jit
def rmsnorm_kernel(
    X, W, Out,
    stride_xz, stride_xm,
    stride_wz,
    stride_oz, stride_om,
    Z, M,
    BLOCK_M: tl.constexpr,
    EPS: tl.constexpr
):
    # Define the program ID
    off_z = tl.program_id(axis=0)
    off_m = tl.program_id(axis=1)

    # Load data from global memory
    off_x = off_z * stride_xz + off_m * stride_xm
    x = tl.load(X + off_x, mask=off_m < M, other=0.0)
    w = tl.load(W + off_z * stride_wz, mask=off_z < Z, other=0.0)

    # Compute the mean of squares
    sum_squares = tl.sum(x * x, axis=0)
    mean_squares = sum_squares / M

    # Compute the RMS norm
    inv_sqrt_mean_squares = 1.0 / tl.sqrt(mean_squares + EPS)
    normalized_x = x * inv_sqrt_mean_squares

    # Scale by the weight
    scaled_x = normalized_x * w

    # Store the result back to global memory
    off_o = off_z * stride_oz + off_m * stride_om
    tl.store(Out + off_o, scaled_x, mask=off_m < M)

def kernel(X, W):
    assert X.is_cuda and W.is_cuda, "Inputs must be CUDA tensors"
    assert X.dtype == torch.float16 and W.dtype == torch.float16, "Inputs must be float16"
    
    # Get the shape of the input tensors
    Z, M = X.shape
    _, Z_W = W.shape
    assert Z == Z_W, "The last dimension of X and the first dimension of W must match"

    # Define the block size
    BLOCK_M = 128

    # Define the grid size
    grid = (triton.cdiv(Z, 1), triton.cdiv(M, BLOCK_M))

    # Launch the Triton kernel
    rmsnorm_kernel[grid](
        X, W, X,
        X.stride(0), X.stride(1),
        W.stride(0),
        X.stride(0), X.stride(1),
        Z, M,
        BLOCK_M=BLOCK_M,
        EPS=1e-6,
        num_warps=4,
        in0=X,
        in1=W,
        out0=X
    )

    return X

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(32, 1024, device='cuda', dtype=torch.float16)
    W = torch.randn(32, device='cuda', dtype=torch.float16)

    # Reference implementation
    ref_out = (X.float() * torch.rsqrt(X.float().pow(2).mean(-1, keepdim=True) + 1e-6) * W.float()).to(X.dtype)

    # Triton implementation
    out = kernel(X.clone(), W.clone())

    # Check if the outputs are the same
    print(torch.allclose(out, ref_out))
