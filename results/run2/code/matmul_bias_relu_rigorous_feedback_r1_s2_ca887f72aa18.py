import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel for matrix multiplication with bias and ReLU activation
@triton.jit
def matmul_bias_relu_kernel(
    A_ptr, B_ptr, Bias_ptr, Out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # Define the position of the thread within the block
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    
    # Compute the starting indices for this block
    off_m = pid // (num_pid_n * num_pid_k) * BLOCK_M
    off_n = (pid % (num_pid_n * num_pid_k)) // num_pid_k * BLOCK_N
    off_k = (pid % (num_pid_n * num_pid_k)) % num_pid_k * BLOCK_K
    
    # Load A, B, and Bias into shared memory
    sA = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float16)
    sB = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float16)
    sBias = tl.zeros((BLOCK_N,), dtype=tl.float16)
    
    # Load A and B into shared memory
    for k in range(0, BLOCK_K, 32):
        if off_m + tl.arange(0, BLOCK_M) < M and off_k + k + tl.arange(0, 32) < K:
            sA[tl.arange(0, BLOCK_M), tl.arange(0, 32)] = tl.load(A_ptr + off_m * stride_am + (off_k + k) * stride_ak, mask=(off_m + tl.arange(0, BLOCK_M) < M) & (off_k + k + tl.arange(0, 32) < K))
        if off_k + k + tl.arange(0, 32) < K and off_n + tl.arange(0, BLOCK_N) < N:
            sB[tl.arange(0, 32), tl.arange(0, BLOCK_N)] = tl.load(B_ptr + (off_k + k) * stride_bk + off_n * stride_bn, mask=(off_k + k + tl.arange(0, 32) < K) & (off_n + tl.arange(0, BLOCK_N) < N))
    
    # Load Bias into shared memory
    if off_n + tl.arange(0, BLOCK_N) < N:
        sBias[tl.arange(0, BLOCK_N)] = tl.load(Bias_ptr + off_n * stride_outn, mask=(off_n + tl.arange(0, BLOCK_N) < N))
    
    # Perform the matrix multiplication and add bias
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, BLOCK_K, 32):
        a = sA[:, k:k+32]
        b = sB[k:k+32, :]
        acc += tl.dot(a, b)
    
    # Add bias
    acc += sBias
    
    # Apply ReLU activation
    acc = tl.where(acc > 0, acc, 0.0)
    
    # Store the result back to global memory
    if off_m + tl.arange(0, BLOCK_M) < M and off_n + tl.arange(0, BLOCK_N) < N:
        tl.store(Out_ptr + off_m * stride_outm + off_n * stride_outn, acc, mask=(off_m + tl.arange(0, BLOCK_M) < M) & (off_n + tl.arange(0, BLOCK_N) < N))

# Define the kernel wrapper function
def kernel(A, B, Bias):
    M, K = A.shape
    _, N = B.shape
    assert K == B.shape[0], "Matrix dimensions do not match"
    assert N == Bias.shape[0], "Bias dimension does not match output dimension"
    
    # Allocate output tensor
    Out = torch.empty(M, N, device='cuda', dtype=torch.float16)
    
    # Define grid size
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    grid = lambda META: (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )
    
    # Launch the kernel
    matmul_bias_relu_kernel[grid](
        A.data_ptr(), B.data_ptr(), Bias.data_ptr(), Out.data_ptr(),
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        Out.stride(0), Out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return Out

# Example usage
if __name__ == "__main__":
    # Create random input tensors
    M, K, N = 1024, 1024, 1024
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    Bias = torch.randn(N, device='cuda', dtype=torch.float16)
    
    # Run the kernel
    Out = kernel(A, B, Bias)
    
    # Verify the result
    ref = torch.relu(A.float() @ B.float() + Bias.float()).to(torch.float16)
    print("Max difference:", torch.max(torch.abs(Out - ref)))
