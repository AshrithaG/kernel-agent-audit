import torch
import triton
import triton.language as tl

@triton.jit
def _transpose_kernel(x_ptr, out_ptr, n_rows, n_cols, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Calculate the row and column indices for this block
    row = tl.program_id(0) * BLOCK_M
    col = tl.program_id(1) * BLOCK_N
    
    # Create a mask to ensure we don't go out of bounds
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows < n_rows) & (cols < n_cols)
    
    # Load the data from the input matrix
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Transpose the data
    x = tl.trans(x)
    
    # Store the transposed data to the output matrix
    tl.store(out_ptr + col * n_rows + rows, x.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    
    # Choose block sizes that fit well with the hardware
    BLOCK_M = 128
    BLOCK_N = 128
    
    # Launch the kernel
    grid = (triton.cdiv(n_rows, BLOCK_M), triton.cdiv(n_cols, BLOCK_N))
    _transpose_kernel[grid](x, out, n_rows, n_cols, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    
    return out
