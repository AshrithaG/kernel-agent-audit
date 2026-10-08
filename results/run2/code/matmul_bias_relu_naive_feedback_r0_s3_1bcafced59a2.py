import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel
@triton.jit
def matmul_bias_relu_kernel(
    a_ptr, b_ptr, bias_ptr, output_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    stride_a_m, stride_a_k,
    stride_b_k, stride_b_n,
    stride_bias_n,
    stride_output_m, stride_output_n
):
    # Define the position of the current thread
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Compute the row and column indices for the current thread
    row = pid // num_pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
    col = pid % num_pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create masks to handle cases where the block size is larger than the matrix dimensions
    mask_m = row < M
    mask_n = col < N
    
    # Initialize the accumulator for the matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Perform the matrix multiplication
    for k in range(0, K, BLOCK_K):
        # Load the current block of A and B into shared memory
        a_block = tl.load(a_ptr + row[:, None] * stride_a_m + (k + tl.arange(0, BLOCK_K)) * stride_a_k, mask=mask_m[:, None] & (k + tl.arange(0, BLOCK_K)) < K, other=0.0)
        b_block = tl.load(b_ptr + (k + tl.arange(0, BLOCK_K))[None, :] * stride_b_k + col * stride_b_n, mask=(k + tl.arange(0, BLOCK_K))[:, None] < K & mask_n, other=0.0)
        
        # Perform the matrix multiplication for the current block
        acc += tl.dot(a_block, b_block)
    
    # Load the bias into shared memory
    bias_block = tl.load(bias_ptr + col * stride_bias_n, mask=mask_n, other=0.0)
    
    # Add the bias to the accumulator
    acc += bias_block
    
    # Apply ReLU activation
    acc = tl.where(acc > 0, acc, 0.0)
    
    # Store the result back to global memory
    tl.store(output_ptr + row[:, None] * stride_output_m + col * stride_output_n, acc, mask=mask_m[:, None] & mask_n)

# Define the wrapper function to call the Triton kernel
def kernel(a, b, bias):
    M, K = a.shape
    K, N = b.shape
    assert a.dtype == torch.float16 and b.dtype == torch.float16 and bias.dtype == torch.float16, "Inputs must be float16 tensors"
    
    # Allocate output tensor
    output = torch.empty((M, N), dtype=torch.float16, device=a.device)
    
    # Define grid size
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    grid = (num_pid_m * num_pid_n,)
    
    # Launch the Triton kernel
    matmul_bias_relu_kernel[grid](
        a.data_ptr(), b.data_ptr(), bias.data_ptr(), output.data_ptr(),
        M, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        bias.stride(0),
        output.stride(0), output.stride(1)
    )
    
    return output

# Example usage
if __name__ == "__main__":
    M, K, N = 1024, 512, 768
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")
    bias = torch.randn(N, dtype=torch.float16, device="cuda")
    
    # Reference implementation
    reference_output = torch.relu(a.float() @ b.float() + bias.float()).to(a.dtype)
    
    # Triton kernel implementation
    triton_output = kernel(a, b, bias)
    
    # Check if the outputs are equal within floating-point tolerance
    assert torch.allclose(reference_output, triton_output, atol=1e-3, rtol=1e-3), "Outputs do not match"
    print("Kernel output matches reference output")
