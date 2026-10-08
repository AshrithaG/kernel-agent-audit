import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_bias_relu(a_ptr, b_ptr, bias_ptr, out_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Define the row and column indices for this program instance
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # Create a range of indices for the K dimension
    k_range = tl.arange(0, BLOCK_K)
    
    # Load the bias into shared memory
    bias = tl.load(bias_ptr + col, mask=col < N, other=0.0).to(tl.float32)
    
    # Initialize the accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Loop over the K dimension
    for k in range(K // BLOCK_K):
        # Load the elements of A and B into shared memory
        a = tl.load(a_ptr + row * K + k * BLOCK_K + k_range, mask=k_range < K, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + k * BLOCK_K + k_range * N + col, mask=k_range < K, other=0.0).to(tl.float32)
        
        # Perform the matrix multiplication and accumulate the result
        acc += tl.dot(a, b)
    
    # Apply the ReLU activation function
    acc = tl.maximum(acc, 0.0)
    
    # Add the bias
    acc += bias
    
    # Store the result back to global memory
    tl.store(out_ptr + row * N + col, acc.to(out_ptr.dtype.element_ty), mask=(row < M) & (col < N))

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
    
    # Launch the kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_bias_relu[(grid[0], grid[1])](a, b, bias, out, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    
    return out
