import torch
import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def bias_gelu_kernel(
    X_ptr, B_ptr, Y_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # Compute the position of this program within the grid
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute the starting row and column for this block
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    # Create a mask for the elements that are within the bounds of the matrix
    mask = (row_start + tl.arange(0, BLOCK_M)) < M

    # Load the bias vector into shared memory
    b = tl.load(B_ptr + tl.arange(0, BLOCK_N), mask=mask[:, None], other=0.0)

    # Load the matrix X into shared memory
    x = tl.load(X_ptr + row_start * N + tl.arange(0, BLOCK_N), mask=mask[:, None], other=0.0)

    # Add the bias vector to each row of the matrix
    x += b

    # Approximate GELU using tanh
    cdf = 0.5 * (1.0 + tl.tanh((x + 0.044715 * x**3) / 1.772454))
    y = x * cdf

    # Store the result back to global memory
    tl.store(Y_ptr + row_start * N + tl.arange(0, BLOCK_N), y, mask=mask[:, None])

# Define the wrapper function to launch the kernel
def kernel(X, B):
    M, N = X.shape
    assert B.shape == (N,)
    assert X.is_cuda and B.is_cuda
    assert X.dtype == torch.float16 and B.dtype == torch.float16

    # Define the block size
    BLOCK_M = 128
    BLOCK_N = 128

    # Allocate output tensor
    Y = torch.empty_like(X)

    # Launch the kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    bias_gelu_kernel[grid](
        X.data_ptr(), B.data_ptr(), Y.data_ptr(),
        M, N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )

    return Y

# Example usage
if __name__ == "__main__":
    # Create random tensors
    X = torch.rand((1024, 512), device='cuda', dtype=torch.float16)
    B = torch.rand((512,), device='cuda', dtype=torch.float16)

    # Run the Triton kernel
    Y_triton = kernel(X, B)

    # Run the reference implementation
    Y_ref = F.gelu(X.float() + B.float(), approximate="tanh").to(X.dtype)

    # Check if the results are the same
    assert torch.allclose(Y_triton, Y_ref, atol=1e-3, rtol=1e-3)
    print("Results match!")
