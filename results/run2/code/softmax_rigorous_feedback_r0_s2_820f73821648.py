import torch
import triton
import triton.language as tl

# Define the Triton kernel for row-wise softmax
@triton.jit
def row_softmax_kernel(
    X,  # Input tensor
    Y,  # Output tensor
    N,  # Number of rows
    M,  # Number of columns
    BLOCK_SIZE_M: tl.constexpr,  # Block size for rows
    BLOCK_SIZE_N: tl.constexpr,  # Block size for columns
):
    # Get the program ID for the current thread
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Calculate the row and column indices for the current thread
    row_idx = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Mask out-of-bounds accesses
    mask = (row_idx < N) & (col_idx < M)

    # Load the input data into shared memory
    x = tl.load(X + row_idx[:, None] * M + col_idx, mask=mask[:, None], other=-float('inf'))

    # Compute the maximum value in each row
    max_val = tl.max(x, axis=1)

    # Subtract the maximum value from each element in the row to avoid overflow
    x = x - max_val[:, None]

    # Compute the exponential of each element
    exp_x = tl.exp(x)

    # Compute the sum of the exponentials in each row
    sum_exp_x = tl.sum(exp_x, axis=1)

    # Divide each element by the sum of the exponentials to get the softmax
    softmax_x = exp_x / sum_exp_x[:, None]

    # Store the result back to global memory
    tl.store(Y + row_idx[:, None] * M + col_idx, softmax_x, mask=mask[:, None])

# Define the kernel wrapper function
def kernel(X):
    # Get the shape of the input tensor
    N, M = X.shape

    # Define the block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128

    # Calculate the number of blocks needed
    num_blocks_m = triton.cdiv(N, BLOCK_SIZE_M)
    num_blocks_n = triton.cdiv(M, BLOCK_SIZE_N)

    # Allocate memory for the output tensor
    Y = torch.empty_like(X)

    # Launch the Triton kernel
    row_softmax_kernel[grid=(num_blocks_m, num_blocks_n), block=(BLOCK_SIZE_M, BLOCK_SIZE_N)](
        X, Y, N, M, BLOCK_SIZE_M, BLOCK_SIZE_N
    )

    return Y

# Reference implementation for testing
def reference(X):
    return torch.softmax(X.float(), dim=-1).to(X.dtype)

# Example usage
if __name__ == "__main__":
    # Create a random input tensor
    X = torch.rand((1024, 512), dtype=torch.float16, device='cuda')

    # Run the Triton kernel
    Y_triton = kernel(X)

    # Run the reference implementation
    Y_ref = reference(X)

    # Check if the results are close
    assert torch.allclose(Y_triton, Y_ref, atol=1e-3, rtol=1e-3)

    print("Triton kernel output matches reference implementation!")
