import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_bias_relu(a_ptr, b_ptr, bias_ptr, out_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Each program instance computes one block of the output matrix
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    # Define the range of elements to compute within this block
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    
    # Create masks to ensure we don't read out of bounds
    rm = rows < M
    cm = cols < N
    km = k < K
    
    # Load the relevant slices of A, B, and Bias into registers
    a = tl.load(a_ptr + m * K + k, mask=rm[:, None] & km[None, :], other=0.0).to(tl.float32)
    b = tl.load(b_ptr + k * N + cols, mask=km[:, None] & cm[None, :], other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=cm, other=0.0).to(tl.float32)
    
    # Compute the dot product for this block
    acc = tl.dot(a, b)
    
    # Add the bias and apply ReLU
    acc += bias
    acc = tl.maximum(acc, 0.0)
    
    # Store the result back to the output matrix
    tl.store(out_ptr + m * N + cols, acc.to(out_ptr.dtype.element_ty), mask=rm[:, None] & cm[None, :])

def kernel(a, b, bias):
    a = a.contiguous()
    b = b.contiguous()
    bias = bias.contiguous()
    out = torch.empty_like(a @ b)
    
    M, K = a.shape
    K, N = b.shape
    
    # Choose block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32
    
    # Launch the kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_bias_relu[grid](a, b, bias, out, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    
    return out
