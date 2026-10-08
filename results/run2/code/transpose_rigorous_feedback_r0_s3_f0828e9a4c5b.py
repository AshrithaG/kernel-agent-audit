import torch
import triton
import triton.language as tl

@triton.jit
def transpose_kernel(input_ptr, output_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Calculate row and column indices
    row = (pid // num_pid_n) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = (pid % num_pid_n) * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load data from input into a tile
    input_tile = tl.load(input_ptr + row[:, None] * N + col, mask=row[:, None] < M & col < N, other=0.0)

    # Transpose the tile
    output_tile = input_tile.T

    # Store the transposed tile to output
    tl.store(output_ptr + col[:, None] * M + row, output_tile, mask=row[:, None] < N & col < M)

def kernel(x):
    M, N = x.shape
    BLOCK_M = 32
    BLOCK_N = 32
    output = torch.empty_like(x).T.contiguous()

    grid = lambda META: (tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N),)
    transpose_kernel[grid](x.data_ptr(), output.data_ptr(), M, N, BLOCK_M, BLOCK_N)
    return output

# Example usage:
# x = torch.randn(1024, 2048, device='cuda', dtype=torch.float16)
# y = kernel(x)
# print(y.shape, y.dtype)
