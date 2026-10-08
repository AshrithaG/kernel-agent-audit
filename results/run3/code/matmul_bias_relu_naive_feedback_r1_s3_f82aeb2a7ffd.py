import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_bias_relu(a_ptr, b_ptr, bias_ptr, out_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Define the block and thread indices
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # Define the range of elements to process
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    k_range = tl.arange(0, BLOCK_K)
    
    # Compute the bounds
    row_bounds = rows < M
    col_bounds = cols < N
    k_bounds = k_range < K
    
    # Load the matrices
    a = tl.load(a_ptr + row * K + k_range, mask=row_bounds[:, None] & k_bounds[None, :], other=0.0).to(tl.float32)
    b = tl.load(b_ptr + k_range * N + cols, mask=k_bounds[:, None] & col_bounds[None, :], other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=col_bounds, other=0.0).to(tl.float32)
    
    # Perform the matrix multiplication and add bias
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_block = tl.load(a_ptr + row * K + k + k_range, mask=row_bounds[:, None] & k_bounds[None, :], other=0.0).to(tl.float32)
        b_block = tl.load(b_ptr + (k + k_range) * N + cols, mask=k_bounds[:, None] & col_bounds[None, :], other=0.0).to(tl.float32)
        acc += tl.dot(a_block, b_block)
    
    acc += bias
    
    # Apply ReLU
    acc = tl.where(acc > 0, acc, 0.0)
    
    # Store the result
    tl.store(out_ptr + row * N + cols, acc.to(out_ptr.dtype.element_ty), mask=row_bounds[:, None] & col_bounds[None, :])

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
    _matmul_bias_relu[grid](a, b, bias, out, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    
    return out
