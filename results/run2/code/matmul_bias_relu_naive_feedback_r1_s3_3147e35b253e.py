import torch
import triton
import triton.language as tl
import math

# Triton kernel for matmul + bias + relu
@triton.jit
def matmul_bias_relu_kernel(
    A, B, Bias, Output,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    stride_a, stride_b, stride_bias, stride_output
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Compute which row/column each program handles
    row = pid // num_pid_n * BLOCK_M
    col = pid % num_pid_n * BLOCK_N

    # Each thread computes a single element in the output matrix
    pid_m = tl.arange(0, BLOCK_M)
    pid_n = tl.arange(0, BLOCK_N)
    rm = row + pid_m
    rn = col + pid_n

    # Load the bias into shared memory
    bias = tl.load(Bias + rn, mask=rn < N, other=0.0)

    # Accumulate the result
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Load elements from A and B into shared memory
        a = tl.load(A + rm[:, None] * stride_a + k, mask=rm[:, None] < M, other=0.0)
        b = tl.load(B + k * stride_b + rn, mask=rn < N, other=0.0)
        acc += tl.dot(a, b)

    # Add bias and apply ReLU
    acc += bias
    acc = tl.where(acc > 0, acc, 0.0)

    # Store the result to the output matrix
    tl.store(Output + rm[:, None] * stride_output + rn, acc, mask=(rm[:, None] < M) & (rn < N))

# Wrapper function to call the Triton kernel
def kernel(A, B, Bias):
    M, K = A.shape
    K, N = B.shape
    assert Bias.shape == (N,), "Bias must have shape (N,)"

    # Allocate output tensor
    Output = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Define grid and block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32
    grid = lambda META: (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    # Launch the Triton kernel
    matmul_bias_relu_kernel[grid](
        A, B, Bias, Output,
        M, K, N,
        BLOCK_M, BLOCK_N, BLOCK_K,
        A.stride(0), B.stride(0), Bias.stride(0), Output.stride(0)
    )

    return Output

# Example usage
if __name__ == "__main__":
    import torch
    torch.manual_seed(0)

    # Create random tensors
    M, K, N = 128, 256, 64
    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')
    Bias = torch.randn(N, dtype=torch.float16, device='cuda')

    # Run the reference implementation
    reference_result = torch.relu(A.float() @ B.float() + Bias.float()).to(A.dtype)

    # Run the Triton kernel
    triton_result = kernel(A, B, Bias)

    # Check if results are close
    assert torch.allclose(reference_result, triton_result, atol=1e-3, rtol=1e-3)

    print("Results match within tolerance.")
