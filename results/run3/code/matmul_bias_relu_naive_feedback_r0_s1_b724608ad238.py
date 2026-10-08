import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_bias_relu(a_ptr, b_ptr, bias_ptr, out_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Get the program ID for the block
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # Define the range of elements to process in this block
    m_range = tl.arange(0, BLOCK_M)
    n_range = tl.arange(0, BLOCK_N)
    k_range = tl.arange(0, BLOCK_K)
    
    # Create masks for bounds checking
    m_mask = m_range < BLOCK_M
    n_mask = n_range < BLOCK_N
    k_mask = k_range < BLOCK_K
    
    # Load the bias into shared memory
    bias = tl.load(bias_ptr + n_range, mask=n_mask, other=0.0).to(tl.float32)
    
    # Initialize accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Perform the matrix multiplication
    for k in range(K // BLOCK_K):
        # Load a block of A and B into shared memory
        a_block = tl.load(a_ptr + row * K + k * BLOCK_K + k_range, mask=k_mask, other=0.0).to(tl.float32)
        b_block = tl.load(b_ptr + col * K + k * BLOCK_K + k_range, mask=k_mask, other=0.0).to(tl.float32)
        
        # Update the accumulators
        acc += tl.dot(a_block[:, None], b_block[None, :])
    
    # Add the bias
    acc += bias
    
    # Apply ReLU
    acc = tl.maximum(acc, 0.0)
    
    # Store the result
    tl.store(out_ptr + row * N + n_range, acc.to(out_ptr.dtype.element_ty), mask=n_mask)

def kernel(a, b, bias):
    a = a.contiguous()
    b = b.contiguous()
    bias = bias.contiguous()
    out = torch.empty_like(a.matmul(b) + bias)
    M, K = a.shape
    K, N = b.shape
    
    # Define block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    # Launch the kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_bias_relu[grid](a, b, bias, out, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    
    return out
