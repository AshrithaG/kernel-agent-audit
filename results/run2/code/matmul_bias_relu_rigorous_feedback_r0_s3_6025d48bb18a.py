import torch
import triton
import triton.language as tl
import math

# Triton kernel for matrix multiplication with bias and ReLU activation
@triton.jit
def matmul_bias_relu_kernel(
    A_ptr, B_ptr, Bias_ptr, Out_ptr,
    M, N, K,
    stride_a, stride_b, stride_bias, stride_out,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACTIVATION: tl.constexpr = True
):
    # Define the program ID
    pid = tl.program_id(axis=0)
    # Calculate the row and column indices
    row = tl.arange(0, BLOCK_M)
    col = tl.arange(0, BLOCK_N)
    # Calculate the offsets for A and B
    a_offset = row[:, None] * stride_a + pid * BLOCK_K
    b_offset = col[None, :] * stride_b + pid * BLOCK_K
    # Load A and B into shared memory
    a = tl.load(A_ptr + a_offset, mask=row[:, None] < M, other=0.0)
    b = tl.load(B_ptr + b_offset, mask=col[None, :] < N, other=0.0)
    # Initialize the accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Perform the matrix multiplication
    for k in range(0, K, BLOCK_K):
        # Update the offsets for A and B
        a_offset = row[:, None] * stride_a + (k + pid) * BLOCK_K
        b_offset = col[None, :] * stride_b + (k + pid) * BLOCK_K
        # Load A and B into shared memory
        a = tl.load(A_ptr + a_offset, mask=row[:, None] < M, other=0.0)
        b = tl.load(B_ptr + b_offset, mask=col[None, :] < N, other=0.0)
        # Perform the dot product
        acc += tl.dot(a, b)
    # Add the bias
    bias_offset = col * stride_bias
    bias = tl.load(Bias_ptr + bias_offset, mask=col < N, other=0.0)
    acc += bias
    # Apply ReLU activation if needed
    if ACTIVATION:
        acc = tl.where(acc > 0, acc, 0.0)
    # Store the result
    out_offset = row[:, None] * stride_out + col[None, :]
    tl.store(Out_ptr + out_offset, acc, mask=row[:, None] < M & col[None, :] < N)

# Wrapper function to call the Triton kernel
def kernel(A, B, Bias):
    # Get the shapes of the input tensors
    M, K = A.shape
    K, N = B.shape
    # Define the grid size
    grid_size = lambda opt: (triton.cdiv(M, opt.BLOCK_M),)
    # Call the Triton kernel
    Out = torch.empty((M, N), device=A.device, dtype=torch.float32)
    matmul_bias_relu_kernel[grid_size](A, B, Bias, Out, M, N, K,
                                       A.stride(0), B.stride(0), Bias.stride(0), Out.stride(0),
                                       BLOCK_M=128, BLOCK_N=128, BLOCK_K=64)
    # Convert the output tensor to the same dtype as the input tensors
    return Out.to(A.dtype)

# Reference implementation for comparison
def reference(A, B, Bias):
    return torch.relu(A.float() @ B.float() + Bias.float()).to(A.dtype)

# Example usage
if __name__ == "__main__":
    # Create example input tensors
    M, K, N = 1024, 512, 768
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    Bias = torch.randn(N, device="cuda", dtype=torch.float16)
    # Call the kernel and reference implementations
    Out_kernel = kernel(A, B, Bias)
    Out_reference = reference(A, B, Bias)
    # Check if the outputs are equal within floating-point tolerance
    assert torch.allclose(Out_kernel, Out_reference, atol=1e-3, rtol=1e-3)
    print("Kernel output matches reference output!")
