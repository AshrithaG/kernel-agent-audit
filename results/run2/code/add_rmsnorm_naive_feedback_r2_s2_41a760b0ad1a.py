import torch
import triton
import triton.language as tl

# Define the Triton kernel
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
):
    pid_z = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    
    # Calculate the position in the grid
    off_z = pid_z * BLOCK_SIZE_Z
    off_m = pid_m * BLOCK_SIZE_M
    
    # Load data from global memory
    Xs = tl.load(X + off_z * stride_xz + off_m * stride_xm, mask=off_z < N, other=0.0)
    Rs = tl.load(R + off_z * stride_rz + off_m * stride_rm, mask=off_z < N, other=0.0)
    Ws = tl.load(W + off_m * stride_wn, mask=off_m < M, other=0.0)
    
    # Compute the sum
    Hs = Xs + Rs
    
    # Compute the mean of squares
    sum_squares = tl.sum(Hs**2, axis=0)
    mean_squares = sum_squares / N
    
    # Compute the root mean square
    rms = tl.sqrt(mean_squares + 1e-6)
    
    # Normalize
    Normed = Hs / rms
    
    # Scale by W
    Outs = Normed * Ws
    
    # Store the result back to global memory
    tl.store(Out + off_z * stride_oz + off_m * stride_on, Outs, mask=off_z < N)

# Define the kernel wrapper
def kernel(X, R, W):
    N, M = X.shape
    assert R.shape == (N, M)
    assert W.shape == (M,)
    
    # Define the block sizes
    BLOCK_SIZE_Z = 128
    BLOCK_SIZE_M = 32
    
    # Allocate output tensor
    Out = torch.empty_like(X)
    
    # Launch the kernel
    grid = (
        triton.cdiv(N, BLOCK_SIZE_Z),
        triton.cdiv(M, BLOCK_SIZE_M),
    )
    rmsnorm_kernel[grid](
        X, R, W, Out,
        X.stride(0), X.stride(1),
        R.stride(0), R.stride(1),
        W.stride(0), W.stride(0),
        Out.stride(0), Out.stride(1),
        N, M,
        BLOCK_SIZE_Z=BLOCK_SIZE_Z,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    
    return Out

# Example usage
if __name__ == "__main__":
    X = torch.randn((1024, 512), device='cuda', dtype=torch.float16)
    R = torch.randn((1024, 512), device='cuda', dtype=torch.float16)
    W = torch.randn((512,), device='cuda', dtype=torch.float16)
    
    out = kernel(X, R, W)
    print(out)
