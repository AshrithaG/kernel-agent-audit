import torch
import triton
import triton.language as tl

@triton.jit
def _transpose(x_ptr, out_ptr, m, n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Calculate the row and column indices for this block
    row = tl.program_id(0) * BLOCK_M
    col = tl.program_id(1) * BLOCK_N
    
    # Create a range for the block
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    
    # Calculate the bounds for the block
    row_mask = rows < m
    col_mask = cols < n
    
    # Load the data from the input matrix
    x = tl.load(x_ptr + row * n + cols, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
    
    # Transpose the data
    x = tl.trans(x)
    
    # Store the transposed data to the output matrix
    tl.store(out_ptr + col * m + rows, x.to(out_ptr.dtype.element_ty), mask=row_mask[:, None] & col_mask[None, :])

def kernel(x):
    x = x.contiguous()
    m, n = x.shape
    out = torch.empty((n, m), dtype=x.dtype, device=x.device)
    
    # Define block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    
    # Calculate the grid size
    grid_m = triton.cdiv(m, BLOCK_M)
    grid_n = triton.cdiv(n, BLOCK_N)
    
    # Launch the kernel
    _transpose[(grid_m, grid_n)](x, out, m, n, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    
    return out
