import torch
import triton
import triton.language as tl

@triton.jit
def matmul_bias_relu_kernel(
    A, B, Bias, Output,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_bias, stride_output
):
    pid = tl.program_id(axis=0)
    tile_m = tl.arange(0, BLOCK_M)
    tile_n = tl.arange(0, BLOCK_N)
    m = pid * BLOCK_M + tile_m
    n = tile_n
    k = tl.arange(0, BLOCK_K)
    
    # Load A and B into registers
    a = tl.load(A + m[:, None] * stride_am + k[None, :] * stride_ak, mask=m[:, None] < M & k[None, :] < K)
    b = tl.load(B + k[:, None] * stride_bk + n[None, :] * stride_bn, mask=k[:, None] < K & n[None, :] < N)
    
    # Compute matrix multiplication
    p = tl.dot(a, b)
    
    # Load bias
    bias = tl.load(Bias + n, mask=n < N)
    
    # Add bias and apply ReLU
    p += bias
    p = tl.where(p > 0, p, 0)
    
    # Store the result
    tl.store(Output + m[:, None] * stride_output + n[None, :], p, mask=m[:, None] < M & n[None, :] < N)

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
        A, B, Bias, Output,
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
    A = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
    B = torch.randn(512, 256, device='cuda', dtype=torch.float16)
    Bias = torch.randn(256, device='cuda', dtype=torch.float16)

    # Reference implementation
    reference_output = torch.relu(A.float() @ B.float() + Bias.float()).to(A.dtype)

    # Triton implementation
    triton_output = kernel(A, B, Bias)

    # Check correctness
    assert torch.allclose(reference_output, triton_output, atol=1e-3, rtol=1e-3)

    print("Output matches reference within tolerance.")
