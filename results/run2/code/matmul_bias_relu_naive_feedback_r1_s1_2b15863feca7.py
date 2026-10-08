import torch
import triton
import triton.language as tl

# Triton kernel for matmul + bias + relu
@triton.jit
def matmul_bias_relu_kernel(
    A_ptr, B_ptr, Bias_ptr, Output_ptr,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    stride_a, stride_b, stride_bias, stride_output
):
    # Compute the row and column indices of the current block
    row = tl.program_id(axis=0) * BLOCK_M
    col = tl.program_id(axis=1) * BLOCK_N

    # Create masks to ensure we don't access out-of-bounds elements
    mask_a = row + tl.arange(0, BLOCK_M) < M
    mask_b = col + tl.arange(0, BLOCK_N) < N

    # Initialize accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Perform the matrix multiplication
    for k in range(K, BLOCK_SIZE=BLOCK_K):
        a = tl.load(A_ptr + row[:, None] * stride_a + k, mask=mask_a[:, None])
        b = tl.load(B_ptr + k * stride_b + col, mask=mask_b[None, :])
        acc += tl.dot(a, b)

    # Load the bias
    bias = tl.load(Bias_ptr + col, mask=mask_b, other=0.0)

    # Apply the bias and ReLU activation
    acc += bias
    acc = tl.maximum(acc, tl.zeros_like(acc))

    # Store the result
    tl.store(Output_ptr + row[:, None] * stride_output + col, acc, mask=mask_a[:, None] & mask_b[None, :])

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
    grid = lambda META: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch the Triton kernel
    matmul_bias_relu_kernel[grid](
        A_ptr=A.data_ptr(),
        B_ptr=B.data_ptr(),
        Bias_ptr=Bias.data_ptr(),
        Output_ptr=Output.data_ptr(),
        M=M, K=K, N=N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        stride_a=A.stride(0),
        stride_b=B.stride(0),
        stride_bias=Bias.stride(0),
        stride_output=Output.stride(0)
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
