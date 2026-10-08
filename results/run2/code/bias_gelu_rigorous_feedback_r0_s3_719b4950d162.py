import torch
import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def add_bias_gelu_kernel(
    X_ptr, B_ptr, Y_ptr,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # Define the indices
    m = tl.program_id(0) * BLOCK_SIZE_M + tl.thread_idx(0)
    n = tl.program_id(1) * BLOCK_SIZE_N + tl.thread_idx(1)

    # Load the data
    X = tl.load(X_ptr + m * N + n, mask=m < M & n < N, other=0.0)
    B = tl.load(B_ptr + n, mask=n < N, other=0.0)

    # Add bias
    X = X + B

    # Tanh approximation of GELU
    C = 0.044715
    X_tanh = tl.tanh(0.79788456 * X * (1 + 0.044715 * X * X))
    Y = 0.5 * X * (1 + X_tanh)

    # Store the result
    tl.store(Y_ptr + m * N + n, Y, mask=m < M & n < N)

# Define the wrapper function
def kernel(X, B):
    M, N = X.shape
    # Define the grid
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    # Launch the kernel
    add_bias_gelu_kernel[grid](
        X.contiguous(), B.contiguous(), X.contiguous(),
        M, N,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=128,
    )
    return X

# Example usage
if __name__ == "__main__":
    X = torch.rand((1024, 512), device='cuda', dtype=torch.float16)
    B = torch.rand((512,), device='cuda', dtype=torch.float16)
    Y = kernel(X, B)
    print(Y)
