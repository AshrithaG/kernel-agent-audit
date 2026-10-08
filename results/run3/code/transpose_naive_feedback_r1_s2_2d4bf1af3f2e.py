import torch
import triton
import triton.language as tl

@triton.jit
def _transpose(x_ptr, out_ptr, m, n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Each block transposes a submatrix of size BLOCK_M x BLOCK_N
    row = tl.program_id(0) * BLOCK_M
    col = tl.program_id(1) * BLOCK_N
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    
    # Create masks to ensure we don't go out of bounds
    row_mask = rows < m - row
    col_mask = cols < n - col
    mask = row_mask[:, None] & col_mask[None, :]
    
    # Load the submatrix from the input
    x = tl.load(x_ptr + row * n + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Transpose the submatrix
    x_transposed = tl.trans(x)
    
    # Store the transposed submatrix in the output
    tl.store(out_ptr + col * m + rows, x_transposed.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x):
    x = x.contiguous()
    m, n = x.shape
    out = torch.empty((n, m), dtype=x.dtype, device=x.device)
    
    # Calculate the number of blocks needed
    BLOCK_M = triton.next_power_of_2(m)
    BLOCK_N = triton.next_power_of_2(n)
    grid_m = (m + BLOCK_M - 1) // BLOCK_M
    grid_n = (n + BLOCK_N - 1) // BLOCK_N
    
    # Launch the kernel
    _transpose[(grid_m, grid_n)](x, out, m, n, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out
