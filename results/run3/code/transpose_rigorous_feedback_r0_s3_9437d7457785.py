import torch
import triton
import triton.language as tl

@triton.jit
def _transpose(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Calculate the row and column indices for this program instance
    row = tl.program_id(0) * BLOCK_M
    col = tl.program_id(1) * BLOCK_N
    
    # Create a range for the block elements
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    
    # Create masks to ensure we don't go out of bounds
    row_mask = rows < M - row
    col_mask = cols < N - col
    
    # Load the data from the input matrix
    x = tl.load(x_ptr + row * N + cols, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
    
    # Transpose the block
    x = tl.trans(x)
    
    # Store the transposed block into the output matrix
    tl.store(out_ptr + col * M + rows, x.to(out_ptr.dtype.element_ty), mask=row_mask[:, None] & col_mask[None, :])

def kernel(x):
    x = x.contiguous()
    M, N = x.shape
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    
    # Define the block size
    BLOCK_M = 32
    BLOCK_N = 32
    
    # Launch the kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _transpose[grid](x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    
    return out
