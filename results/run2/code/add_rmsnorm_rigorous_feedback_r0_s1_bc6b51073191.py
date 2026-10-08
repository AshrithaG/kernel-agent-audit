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
    # Define the position in the grid
    off_z = tl.program_id(axis=0)
    off_m = tl.program_id(axis=1)

    # Load data from global memory
    X = tl.load(X + off_z * stride_xz + off_m * stride_xm, mask=(off_m < M), other=0.0)
    R = tl.load(R + off_z * stride_rz + off_m * stride_rm, mask=(off_m < M), other=0.0)
    W = tl.load(W + off_m * stride_wn, mask=(off_m < M), other=0.0)

    # Compute the sum of squares
    sum_squares = tl.sum(X * X + R * R, axis=0)

    # Compute the RMSNorm factor
    rmsnorm_factor = tl.sqrt(sum_squares / M + EPS)

    # Compute the output
    Out = (X + R) / rmsnorm_factor * W

    # Store the result back to global memory
    tl.store(Out + off_z * stride_oz + off_m * stride_on, Out, mask=(off_m < M))

# Triton kernel wrapper function
def kernel(X, R, W):
    # Get the shape of the input tensors
    B, M = X.shape

    # Define the grid and block sizes
    BLOCK_SIZE_Z = 128
    BLOCK_SIZE_M = 32

    # Define the strides
    stride_xz = X.stride(0)
    stride_xm = X.stride(1)
    stride_rz = R.stride(0)
    stride_rm = R.stride(1)
    stride_wz = W.stride(0)
    stride_wn = W.stride(1)
    stride_oz = Out.stride(0)
    stride_on = Out.stride(1)

    # Allocate the output tensor
    Out = torch.empty_like(X, dtype=torch.float32)

    # Launch the Triton kernel
    grid = (triton.cdiv(B, BLOCK_SIZE_Z), triton.cdiv(M, BLOCK_SIZE_M))
    rmsnorm_kernel[grid](
        X, R, W, Out,
        stride_xz, stride_xm,
        stride_rz, stride_rm,
        stride_wz, stride_wn,
        stride_oz, stride_on,
        B, M,
        BLOCK_SIZE_Z=BLOCK_SIZE_Z,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        EPS=1e-6
    )

    # Convert the output tensor back to the original dtype
    return Out.to(X.dtype)

# Example usage
if __name__ == "__main__":
    # Create example input tensors
    X = torch.randn(128, 512, device='cuda', dtype=torch.float16)
    R = torch.randn(128, 512, device='cuda', dtype=torch.float16)
    W = torch.randn(512, device='cuda', dtype=torch.float16)

    # Call the Triton kernel
    Y = kernel(X, R, W)

    # Print the shape and dtype of the output tensor
    print(Y.shape, Y.dtype)
