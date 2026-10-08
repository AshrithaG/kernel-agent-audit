import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel for matmul, bias addition, and ReLU activation
@triton.jit
def matmul_bias_relu_kernel(
    A_ptr, B_ptr, Bias_ptr, Out_ptr,
    M, N, K,
    stride_a, stride_b, stride_bias, stride_out,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Define the thread identifiers
    pid = tl.program_id(axis=0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    m = pid // grid_n * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid % grid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load the bias into shared memory
    bias = tl.load(Bias_ptr + n, mask=n < N, other=0.0)

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Perform the matrix multiplication
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr + m[:, None] * stride_a + k, mask=m[:, None] < M, other=0.0)
        b = tl.load(B_ptr + k * stride_b + n[None, :], mask=n[None, :] < N, other=0.0)
        acc += tl.dot(a, b)

    # Add the bias and apply ReLU
    acc += bias
    acc = tl.maximum(acc, 0.0)

    # Store the result
    tl.store(Out_ptr + m[:, None] * stride_out + n[None, :], acc, mask=m[:, None] < M & n[None, :] < N)

# Define the wrapper function to call the Triton kernel
def kernel(A, B, Bias):
    # Get the dimensions of the input tensors
    M, K = A.shape
    K, N = B.shape
    assert K == Bias.shape[0], "Bias must have the same length as the number of columns in B"

    # Define the block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_SIZE_M = 8

    # Calculate the grid size
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # Launch the Triton kernel
    matmul_bias_relu_kernel[
        grid_m * grid_n,
        BLOCK_M * BLOCK_N,
        {
            'BLOCK_M': BLOCK_M,
            'BLOCK_N': BLOCK_N,
            'BLOCK_K': BLOCK_K,
            'GROUP_SIZE_M': GROUP_SIZE_M
        }
    ](
        A.data_ptr(), B.data_ptr(), Bias.data_ptr(), A.data_ptr(),
        M, N, K,
        A.stride(0), B.stride(0), Bias.stride(0), A.stride(0),
    )

    # Return the modified A tensor
    return A

# Example usage
if __name__ == "__main__":
    import time

    # Create random input tensors
    M, K, N = 1024, 512, 768
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    Bias = torch.randn(N, device='cuda', dtype=torch.float16)

    # Measure the performance of the reference implementation
    start_time = time.time()
    reference_output = torch.relu(A.float() @ B.float() + Bias.float()).to(A.dtype)
    reference_time = time.time() - start_time
    print(f"Reference time: {reference_time:.6f} seconds")

    # Measure the performance of the Triton kernel
    start_time = time.time()
    kernel_output = kernel(A.clone(), B.clone(), Bias.clone())
    kernel_time = time.time() - start_time
    print(f"Triton kernel time: {kernel_time:.6f} seconds")

    # Verify the correctness of the Triton kernel output
    assert torch.allclose(reference_output, kernel_output), "The Triton kernel output is incorrect"
    print("The Triton kernel output is correct")
