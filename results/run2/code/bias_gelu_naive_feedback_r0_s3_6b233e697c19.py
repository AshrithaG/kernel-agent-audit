import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel
@triton.jit
def bias_gelu_kernel(
    X,  # input matrix
    B,  # bias vector
    Y,  # output matrix
    M,  # number of rows
    N,  # number of columns
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    # Calculate the row and column indices
    row_idx = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_idx = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # Mask out-of-bounds elements
    mask = (row_idx < M) & (col_idx < N)
    # Load the input matrix and bias vector
    x = tl.load(X + row_idx[:, None] * N + col_idx[None, :], mask=mask[:, None])
    b = tl.load(B + col_idx, mask=col_idx < N)
    # Add the bias vector to each row of the input matrix
    x += b
    # Apply the tanh approximation of GELU
    cdf = 0.5 * (1.0 + tl.tanh((x + 0.044715 * x**3) / math.sqrt(2.0)))
    y = x * cdf
    # Store the result in the output matrix
    tl.store(Y + row_idx[:, None] * N + col_idx[None, :], y, mask=mask[:, None])

# Define the wrapper function
def kernel(X, B):
    M, N = X.shape
    # Define the block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    # Calculate the grid size
    grid_size = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    # Allocate the output tensor
    Y = torch.empty_like(X)
    # Launch the kernel
    bias_gelu_kernel[
        grid_size,
        {
            "BLOCK_SIZE_M": BLOCK_SIZE_M,
            "BLOCK_SIZE_N": BLOCK_SIZE_N,
        },
    ](
        X, B, Y, M, N
    )
    return Y

# Example usage
if __name__ == "__main__":
    # Create random input tensors
    X = torch.rand((1024, 512), device="cuda", dtype=torch.float16)
    B = torch.rand((512,), device="cuda", dtype=torch.float16)
    # Run the kernel
    Y = kernel(X, B)
    print(Y)
