import torch
import triton
import triton.language as tl
import math

# Triton kernel for SwiGLU activation
@triton.jit
def swiglu_kernel(
    X, Y,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    X_STRIDE_0: tl.constexpr,
    X_STRIDE_1: tl.constexpr,
    Y_STRIDE_0: tl.constexpr,
    Y_STRIDE_1: tl.constexpr,
):
    # Get the thread ID
    idx = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)

    # Calculate the row and column indices
    row = idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = pid * N

    # Load the first half and second half of the input
    x_first_half = tl.load(X + row[:, None] * X_STRIDE_0 + col[None, :] * X_STRIDE_1, mask=row[:, None] < X.shape[0], other=0.0)
    x_second_half = tl.load(X + row[:, None] * X_STRIDE_0 + (col[None, :] + N) * X_STRIDE_1, mask=row[:, None] < X.shape[0], other=0.0)

    # Apply SiLU to the first half
    x_silu = x_first_half * tl.sigmoid(x_first_half)

    # Multiply by the second half
    y = x_silu * x_second_half

    # Store the result
    tl.store(Y + row[:, None] * Y_STRIDE_0 + col[None, :] * Y_STRIDE_1, y, mask=row[:, None] < X.shape[0])

# Wrapper function to launch the Triton kernel
def kernel(X):
    N = X.shape[-1] // 2
    BLOCK_SIZE = 128
    grid = lambda META: (math.ceil(X.shape[0] / META['BLOCK_SIZE']), X.shape[1])
    Y = torch.empty_like(X)
    swiglu_kernel[grid](X, Y, N=N, BLOCK_SIZE=BLOCK_SIZE, 
                         X_STRIDE_0=X.stride(0), X_STRIDE_1=X.stride(1), 
                         Y_STRIDE_0=Y.stride(0), Y_STRIDE_1=Y.stride(1))
    return Y

# Example usage
if __name__ == "__main__":
    X = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
    Y = kernel(X)
    print(Y.shape, Y.dtype)
