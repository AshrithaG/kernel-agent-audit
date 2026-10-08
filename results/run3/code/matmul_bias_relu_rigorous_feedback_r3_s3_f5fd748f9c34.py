import torch
import triton
import triton.language as tl

# Triton kernel for matmul + bias + relu
@triton.jit
def _matmul_bias_relu(a_ptr, b_ptr, bias_ptr, out_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Determine the row and column indices for this block
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # Create a range for the elements in the block
    m_range = tl.arange(0, BLOCK_M)
    n_range = tl.arange(0, BLOCK_N)
    k_range = tl.arange(0, BLOCK_K)
    
    # Compute the bounds for the current block
    m_bound = min(row * BLOCK_M + BLOCK_M, M)
    n_bound = min(col * BLOCK_N + BLOCK_N, N)
    k_bound = K
    
    # Initialize accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Perform the matrix multiplication
    for k in range(0, k_bound, BLOCK_K):
        # Load a block of A and B into registers
        a_block = tl.load(a_ptr + row * K + k_range, mask=m_range[:, None] < (m_bound - row * BLOCK_M), other=0.0).to(tl.float32)
        b_block = tl.load(b_ptr + k * N + n_range, mask=k_range[None, :] < (k_bound - k), other=0.0).to(tl.float32)
        
        # Perform the dot product
        acc += tl.dot(a_block, b_block)
    
    # Load the bias
    bias_block = tl.load(bias_ptr + n_range, mask=n_range < (n_bound - col * BLOCK_N), other=0.0).to(tl.float32)
    
    # Add the bias and apply ReLU
    acc += bias_block
    acc = tl.maximum(acc, tl.zeros(acc.shape, dtype=tl.float32))
    
    # Store the result
    tl.store(out_ptr + row * N + n_range, acc, mask=n_range < (n_bound - col * BLOCK_N))

def kernel(a, b, bias):
    a = a.contiguous()
    b = b.contiguous()
    bias = bias.contiguous()
    out = torch.empty_like(a.matmul(b))
    
    M, K = a.shape
    K, N = b.shape
    
    # Define the block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32
    
    # Launch the Triton kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_bias_relu[(grid[0], grid[1])](a, b, bias, out, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    
    return out
