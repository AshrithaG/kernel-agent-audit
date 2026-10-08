import torch
import triton
import triton.language as tl

# Define the Triton kernel for row-wise softmax
@triton.jit
def row_softmax_kernel(
    X,  # input matrix
    Y,  # output matrix
    stride_xm,  # stride of X matrix
    stride_ym,  # stride of Y matrix
    N,  # number of rows
    M,  # number of columns
    BLOCK_SIZE_M: tl.constexpr,  # number of rows per block
    BLOCK_SIZE_N: tl.constexpr  # number of columns per block
):
    # Get the row and column indices
    row_idx = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_idx = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Load the data into shared memory
    x = tl.load(X + row_idx[:, None] * stride_xm + col_idx[None, :] * stride_ym, mask=row_idx[:, None] < N & col_idx[None, :] < M, other=0.0)

    # Compute the maximum value in each row
    max_val = tl.max(x, axis=1)
    max_val = tl.broadcast_to(max_val[:, None], x.shape)
    x = x - max_val

    # Compute the exponential of the shifted values
    exp_x = tl.exp(x)

    # Compute the sum of the exponentials in each row
    sum_exp_x = tl.sum(exp_x, axis=1)
    sum_exp_x = tl.broadcast_to(sum_exp_x[:, None], x.shape)

    # Compute the softmax values
    softmax_x = exp_x / sum_exp_x

    # Store the results back to global memory
    tl.store(Y + row_idx[:, None] * stride_ym + col_idx[None, :] * stride_xm, softmax_x, mask=row_idx[:, None] < N & col_idx[None, :] < M)


# Define the wrapper function for the Triton kernel
def kernel(X):
    # Get the shape of the input tensor
    N, M = X.shape

    # Define the block size
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 128

    # Allocate the output tensor
    Y = torch.empty_like(X)

    # Launch the Triton kernel
    grid = (triton.cdiv(N, BLOCK_SIZE_M), triton.cdiv(M, BLOCK_SIZE_N))
    row_softmax_kernel[grid](
        X,
        Y,
        X.stride(0),
        Y.stride(0),
        N,
        M,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )

    return Y


# Example usage
if __name__ == "__main__":
    # Create a random input tensor
    X = torch.rand((1024, 1024), device='cuda', dtype=torch.float16)

    # Compute the softmax using the Triton kernel
    Y_triton = kernel(X)

    # Compute the softmax using PyTorch
    Y_torch = torch.softmax(X, dim=-1)

    # Check if the results are the same
    assert torch.allclose(Y_triton, Y_torch, atol=1e-3, rtol=1e-3)

    print("Triton kernel produces the same result as PyTorch")
