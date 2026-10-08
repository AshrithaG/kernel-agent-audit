import torch
import triton
import triton.language as tl

@triton.jit
def matmul_bias_relu_kernel(
    A_ptr, B_ptr, Bias_ptr, Output_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_bias, stride_output
):
    # Get program ID
    pid = tl.program_id(axis=0)
    # Calculate row and column indices
    row = tl.arange(0, BLOCK_M) + pid * BLOCK_M
    col = tl.arange(0, BLOCK_N)
    # Load A and B into registers
    a = tl.load(A_ptr + row[:, None] * stride_am + col[None, :] * stride_ak, mask=row[:, None] < M & col[None, :] < K)
    b = tl.load(B_ptr + col[:, None] * stride_bk + row[None, :] * stride_bn, mask=row[:, None] < K & col[None, :] < N)
    # Compute matrix multiplication
    p = tl.dot(a, b)
    # Load bias
    bias = tl.load(Bias_ptr + col, mask=col < N)
    # Add bias and apply ReLU
    p += bias
    p = tl.where(p > 0, p, 0)
    # Store the result
    tl.store(Output_ptr + row[:, None] * stride_output + col[None, :], p, mask=row[:, None] < M & col[None, :] < N)

def kernel(A, B, Bias):
    M, K = A.shape
    K, N = B.shape
    assert Bias.shape == (N,), "Bias must have shape (N,)"

    # Define grid size
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid_size = (grid_m,)

    # Allocate output tensor
    Output = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Launch the kernel
    matmul_bias_relu_kernel[grid_size](
        A.data_ptr(), B.data_ptr(), Bias.data_ptr(), Output.data_ptr(),
        M, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        Bias.stride(0), Output.stride(0)
    )

    return Output

# Example usage
if __name__ == "__main__":
    import torch

    # Create random tensors
    A = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    B = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    Bias = torch.randn(1024, device='cuda', dtype=torch.float16)

    # Reference implementation
    reference_output = torch.relu(A.float() @ B.float() + Bias.float()).to(A.dtype)

    # Triton implementation
    triton_output = kernel(A, B, Bias)

    # Check correctness
    assert torch.allclose(reference_output, triton_output, atol=1e-3, rtol=1e-3)

    print("Output matches reference within tolerance.")
