import torch
import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def matmul_bias_relu_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    stride_a1, stride_a2,
    stride_b1, stride_b2,
    stride_bias,
    stride_c1, stride_c2
):
    # Define the thread indices
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Compute the row and column indices for the current thread
    row = pid // num_pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
    col = pid % num_pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create masks to handle edge cases where M or N is not a multiple of BLOCK_M or BLOCK_N
    mask_m = row < M
    mask_n = col < N

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Perform the matrix multiplication
    for k in range(K):
        a = tl.load(A_ptr + row[:, None] * stride_a1 + k * stride_a2, mask=mask_m[:, None], other=0.0)
        b = tl.load(B_ptr + k * stride_b1 + col * stride_b2, mask=mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)

    # Load the bias
    bias = tl.load(Bias_ptr + col, mask=mask_n, other=0.0)

    # Add the bias and apply ReLU
    acc += bias
    acc = tl.where(acc > 0, acc, 0.0)

    # Store the result
    tl.store(C_ptr + row[:, None] * stride_c1 + col * stride_c2, acc, mask=mask_m[:, None])

# Define the wrapper function
def kernel(A, B, Bias):
    M, K = A.shape
    K, N = B.shape
    assert A.dtype == torch.float16 and B.dtype == torch.float16 and Bias.dtype == torch.float16, "Inputs must be float16"
    assert Bias.shape == (N,), "Bias must be of shape (N,)"

    # Allocate the output tensor
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Define the grid size
    BLOCK_M = 16
    BLOCK_N = 16
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    grid_size = (num_pid_m * num_pid_n,)

    # Launch the kernel
    matmul_bias_relu_kernel[grid_size](
        A.data_ptr(), B.data_ptr(), Bias.data_ptr(), C.data_ptr(),
        M, N, K,
        BLOCK_M, BLOCK_N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        Bias.stride(0),
        C.stride(0), C.stride(1)
    )

    return C

# Example usage
if __name__ == "__main__":
    # Create random input tensors
    M, K, N = 128, 256, 512
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    Bias = torch.randn(N, device='cuda', dtype=torch.float16)

    # Run the kernel
    C = kernel(A, B, Bias)

    # Print the result
    print(C)
