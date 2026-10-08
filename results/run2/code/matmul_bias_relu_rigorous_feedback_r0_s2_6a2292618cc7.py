import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, bias_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    
    # Compute each element of C in a block
    m = (pid // num_pid_n // num_pid_k) * BLOCK_M + tl.arange(0, BLOCK_M)
    n = (pid // num_pid_k % num_pid_n) * BLOCK_N + tl.arange(0, BLOCK_N)
    k = (pid % num_pid_k) * BLOCK_K + tl.arange(0, BLOCK_K)
    
    # Load A and B into shared memory
    a_block = tl.load(a_ptr + m[:, None] * K + k[None, :], mask=m[:, None] < M & k[None, :] < K, other=0.0)
    b_block = tl.load(b_ptr + k[:, None] * N + n[None, :], mask=k[:, None] < K & n[None, :] < N, other=0.0)
    
    # Compute matrix multiplication
    p = tl.dot(a_block, b_block)
    
    # Add bias
    bias_block = tl.load(bias_ptr + n, mask=n < N, other=0.0)
    p += bias_block
    
    # Apply ReLU
    p = tl.where(p > 0, p, 0.0)
    
    # Store the result
    tl.store(c_ptr + m[:, None] * N + n[None, :], p, mask=m[:, None] < M & n[None, :] < N)

def kernel(a, b, bias):
    M, K = a.shape
    _, N = b.shape
    
    # Define grid size
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    grid = (num_pid_m * num_pid_n * num_pid_k,)
    
    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Launch kernel
    matmul_kernel[grid](a.contiguous(), b.contiguous(), bias.contiguous(), c.contiguous(), M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    
    return c

# Example usage
if __name__ == "__main__":
    a = torch.randn(128, 512, device='cuda', dtype=torch.float16)
    b = torch.randn(512, 256, device='cuda', dtype=torch.float16)
    bias = torch.randn(256, device='cuda', dtype=torch.float16)
    
    result = kernel(a, b, bias)
    print(result)
