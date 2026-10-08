import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_bias_relu(a_ptr, b_ptr, bias_ptr, out_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Define the block and element indices
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)

    # Load the relevant slices of A, B, and bias
    a = tl.load(a_ptr + row[:, None] * K + k[None, :], mask=row[:, None] < M & k[None, :] < K, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + k[:, None] * N + col[None, :], mask=k[:, None] < K & col[None, :] < N, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + col, mask=col < N, other=0.0).to(tl.float32)

    # Perform the matrix multiplication and add bias
    acc = tl.dot(a, b)
    acc += bias[None, :]

    # Apply ReLU
    acc = tl.where(acc > 0, acc, 0.0)

    # Store the result
    tl.store(out_ptr + row[:, None] * N + col[None, :], acc, mask=row[:, None] < M & col[None, :] < N)

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
