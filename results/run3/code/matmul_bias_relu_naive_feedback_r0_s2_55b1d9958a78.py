import torch
import triton
import triton.language as tl

# Define the Triton kernel for matmul, bias addition, and ReLU activation
@triton.jit
def _matmul_bias_relu(a_ptr, b_ptr, bias_ptr, out_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Get the program ID for the block
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the row and column indices for the current block
    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize the accumulator for the matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Perform the matrix multiplication
    for k in range(K // BLOCK_K):
        # Load the current block of A and B
        a = tl.load(a_ptr + row[:, None] * K + (k * BLOCK_K + tl.arange(0, BLOCK_K))[None, :], mask=row[:, None] < M, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + (k * BLOCK_K + tl.arange(0, BLOCK_K))[None, :] * N + col[None, :], mask=(k * BLOCK_K + tl.arange(0, BLOCK_K))[None, :] < K, other=0.0).to(tl.float32)

        # Perform the dot product
        acc += tl.dot(a, b)

    # Load the bias
    bias = tl.load(bias_ptr + col, mask=col < N, other=0.0).to(tl.float32)

    # Add the bias to the accumulator
    acc += bias

    # Apply ReLU activation
    acc = tl.maximum(acc, 0.0)

    # Store the result
    tl.store(out_ptr + row[:, None] * N + col[None, :], acc, mask=row[:, None] < M & col[None, :] < N)

# Define the wrapper function for the kernel
def kernel(a, b, bias):
    # Ensure the inputs are contiguous
    a = a.contiguous()
    b = b.contiguous()
    bias = bias.contiguous()

    # Get the dimensions of the input tensors
    M, K = a.shape
    K, N = b.shape

    # Allocate the output tensor
    out = torch.empty((M, N), dtype=a.dtype, device=a.device)

    # Define the block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # Launch the kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_bias_relu[(grid[0], grid[1])](a, b, bias, out, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)

    return out
