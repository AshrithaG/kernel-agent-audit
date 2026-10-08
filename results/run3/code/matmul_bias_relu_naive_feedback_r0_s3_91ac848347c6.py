import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_bias_relu(a_ptr, b_ptr, bias_ptr, out_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Determine the row and column indices for this block
    row = tl.program_id(0) * BLOCK_M
    col = tl.program_id(1) * BLOCK_N
    
    # Create a range for the block size
    pid_m = tl.arange(0, BLOCK_M)
    pid_n = tl.arange(0, BLOCK_N)
    pid_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Perform the matrix multiplication
    for k in range(K // BLOCK_K):
        # Load A and B blocks
        a_block = tl.load(a_ptr + row * K + pid_k + k * BLOCK_K * K, mask=(pid_m[:, None] < BLOCK_M) & (pid_k[None, :] < BLOCK_K), other=0.0).to(tl.float32)
        b_block = tl.load(b_ptr + col * K + pid_k + k * BLOCK_K * K, mask=(pid_k[:, None] < BLOCK_K) & (pid_n[None, :] < BLOCK_N), other=0.0).to(tl.float32)
        
        # Matrix multiply
        acc += tl.dot(a_block, b_block)
    
    # Load bias
    bias_block = tl.load(bias_ptr + col + pid_n, mask=pid_n < BLOCK_N, other=0.0).to(tl.float32)
    
    # Add bias and apply ReLU
    acc += bias_block
    acc = tl.where(acc > 0.0, acc, 0.0)
    
    # Store the result
    tl.store(out_ptr + row * N + col + pid_m[:, None] * N + pid_n[None, :], acc, mask=(pid_m[:, None] < BLOCK_M) & (pid_n[None, :] < BLOCK_N))

def kernel(a, b, bias):
    a = a.contiguous()
    b = b.contiguous()
    bias = bias.contiguous()
    out = torch.empty_like(a.matmul(b))
    
    M, K = a.shape
    K, N = b.shape
    
    # Define block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32
    
    # Launch the kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_bias_relu[(grid[0], grid[1])](a, b, bias, out, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    
    return out
