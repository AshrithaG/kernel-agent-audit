import torch
import triton
import triton.language as tl

@triton.jit
def _transpose(x_ptr, out_ptr, m, n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Calculate the row and column indices for this block
    row = tl.program_id(0) * BLOCK_M
    col = tl.program_id(1) * BLOCK_N
    
    # Create a range of indices for the block
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    
    # Create masks to ensure we don't go out of bounds
    mask = (rows < m) & (cols < n)
    
    # Load the data from the input matrix
    x = tl.load(x_ptr + row * n + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Transpose the data
    x_transposed = tl.trans(x)
    
    # Store the transposed data into the output matrix
    tl.store(out_ptr + col * m + rows, x_transposed.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x):
    x = x.contiguous()
    out = torch.empty((x.shape[1], x.shape[0]), dtype=x.dtype, device=x.device)
    m, n = x.shape
    BLOCK_M = 32
    BLOCK_N = 32
    grid_m = triton.cdiv(m, BLOCK_M)
    grid_n = triton.cdiv(n, BLOCK_N)
    _transpose[(grid_m, grid_n)](x, out, m, n, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out
