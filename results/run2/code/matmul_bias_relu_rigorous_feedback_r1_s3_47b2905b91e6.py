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
    # Each thread computes one element of the output matrix
    pid = tl.program_id(axis=0)
    num_threads = tl.num_programs(axis=0)
    
    # Compute the row and column index for this thread
    row = (pid // (N // BLOCK_N)) * BLOCK_M
    col = (pid % (N // BLOCK_N)) * BLOCK_N
    
    # Load A and B into registers
    a = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float16)
    b = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float16)
    
    # Perform the matrix multiplication
    for k in range(K // BLOCK_K):
        a_block_ptr = A_ptr + row * stride_am + (k * BLOCK_K) * stride_ak
        b_block_ptr = B_ptr + (k * BLOCK_K) * stride_bk + col * stride_bn
        
        a += tl.load(a_block_ptr.to(tl.uint16), mask=(row + tl.arange(0, BLOCK_M)) < M & (k * BLOCK_K + tl.arange(0, BLOCK_K)) < K)
        b += tl.load(b_block_ptr.to(tl.uint16), mask=(k * BLOCK_K + tl.arange(0, BLOCK_K)) < K & (col + tl.arange(0, BLOCK_N)) < N)
    
    # Compute the dot product
    p = tl.dot(a, b)
    
    # Load bias
    bias_ptr = Bias_ptr + col
    bias = tl.load(bias_ptr, mask=(col + tl.arange(0, BLOCK_N)) < N)
    
    # Add bias and apply ReLU
    p += bias
    p = tl.where(p > 0, p, 0)
    
    # Store the result
    output_ptr = Output_ptr + row * stride_output + col
    tl.store(output_ptr, p, mask=(row + tl.arange(0, BLOCK_M)) < M & (col + tl.arange(0, BLOCK_N)) < N)

def kernel(A, B, Bias):
    M, K = A.shape
    K, N = B.shape
    assert Bias.shape == (N,), "Bias must have shape (N,)"

    # Define grid size
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    grid_size = (M * N // (BLOCK_M * BLOCK_N),)

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
