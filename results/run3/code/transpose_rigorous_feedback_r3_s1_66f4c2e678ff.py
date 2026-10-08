import torch
import triton
import triton.language as tl

@triton.jit
def _transpose_kernel(in_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Calculate the block and element indices
    row = tl.program_id(0) * BLOCK_M
    col = tl.program_id(1) * BLOCK_N
    
    # Create a range for the elements within the block
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    
    # Mask to ensure we don't go out of bounds
    mask = (rows < M)[:, None] & (cols < N)[None, :]
    
    # Load the input data
    x = tl.load(in_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Transpose the data
    y = tl.trans(x)
    
    # Store the transposed data
    tl.store(out_ptr + col * M + rows, y, mask=mask)

def kernel(x):
    x = x.contiguous()
    M, N = x.shape
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    
    # Define the block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    
    # Calculate the grid size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Launch the kernel
    _transpose_kernel[grid](x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    
    return out
