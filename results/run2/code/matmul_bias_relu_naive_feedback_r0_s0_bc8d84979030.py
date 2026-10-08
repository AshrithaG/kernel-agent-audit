import torch
import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def matmul_bias_relu_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    stride_a_m, stride_a_k,
    stride_b_k, stride_b_n,
    stride_bias_n,
    stride_c_m, stride_c_n
):
    # Define the block indices
    pid = tl.program_id(axis=0)
    bid_m = tl.program_id(axis=1)
    bid_n = tl.program_id(axis=2)

    # Compute the row and column indices for the current thread block
    rm = bid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = bid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load the bias vector
    bias = tl.load(Bias_ptr + rn * stride_bias_n, mask=rn < N)

    # Initialize the accumulator to zero
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Perform the matrix multiplication
    for k in range(K // BLOCK_K):
        # Load the submatrix of A and B
        a = tl.load(A_ptr + rm[:, None] * stride_a_m + (k * BLOCK_K + tl.arange(0, BLOCK_K)) * stride_a_k, mask=rm[:, None] < M)
        b = tl.load(B_ptr + (k * BLOCK_K + tl.arange(0, BLOCK_K))[None, :] * stride_b_k + rn * stride_b_n, mask=rn[None, :] < N)

        # Accumulate the result
        acc += tl.dot(a, b)

    # Add the bias and apply ReLU
    acc = acc + bias
    acc = tl.where(acc > 0, acc, 0)

    # Store the result back to global memory
    tl.store(C_ptr + rm[:, None] * stride_c_m + rn * stride_c_n, acc, mask=(rm[:, None] < M) & (rn < N))

# Define the wrapper function to call the Triton kernel
def matmul_bias_relu(a, b, bias):
    M, K = a.shape
    K, N = b.shape
    assert bias.shape == (N,), "Bias must be a 1D tensor with length N"

    # Define the block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32

    # Allocate the output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Define the grid size
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
        M // BLOCK_M
    )

    # Launch the Triton kernel
    matmul_bias_relu_kernel[grid](
        a.data_ptr(), b.data_ptr(), bias.data_ptr(), c.data_ptr(),
        M, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        bias.stride(0),
        c.stride(0), c.stride(1)
    )

    return c

# Example usage
if __name__ == "__main__":
    # Create some random input tensors
    M, K, N = 128, 256, 512
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    bias = torch.randn(N, device='cuda', dtype=torch.float16)

    # Call the Triton kernel
    c = matmul_bias_relu(a, b, bias)

    # Print the result
    print(c)
