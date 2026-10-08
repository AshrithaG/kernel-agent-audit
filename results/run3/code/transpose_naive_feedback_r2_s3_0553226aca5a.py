import torch
import triton
import triton.language as tl

@triton.jit
def _transpose(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Calculate the position of the current block
    row = tl.program_id(0) * BLOCK_M
    col = tl.program_id(1) * BLOCK_N
    
    # Define the range of elements to process
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    
    # Create masks to ensure we don't go out of bounds
    row_mask = rows < M - row
    col_mask = cols < N - col
    mask = row_mask[:, None] & col_mask[None, :]
    
    # Load the submatrix from the input
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Transpose the submatrix
    x_transposed = tl.trans(x)
    
    # Store the transposed submatrix in the output
    tl.store(out_ptr + col * M + rows, x_transposed.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x):
    x = x.contiguous()
    M, N = x.shape
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    
    # Calculate the number of blocks needed
    BLOCK_M = triton.next_power_of_2(M)
    BLOCK_N = triton.next_power_of_2(N)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    # Launch the kernel
    _transpose[(grid_m, grid_n)](x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out
