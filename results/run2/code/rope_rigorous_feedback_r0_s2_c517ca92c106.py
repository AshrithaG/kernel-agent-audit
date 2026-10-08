import torch
import triton
import triton.language as tl
import math

@triton.jit
def _rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, tokens, heads, dim, BLOCK_SIZE: tl.constexpr):
    # Get the block index and thread index
    bid = tl.program_id(axis=0)
    tid = tl.program_id(axis=1)

    # Calculate the row and column indices
    row = bid * BLOCK_SIZE + tid
    col = tl.load(x_ptr + row * dim, mask=row < tokens, other=0.0)
    
    # Calculate the half dimension
    d = dim // 2
    
    # Load the relevant parts of x, cos, and sin
    x1 = tl.load(x_ptr + row * dim + col * heads * dim, mask=row < tokens, other=0.0)
    x2 = tl.load(x_ptr + row * dim + col * heads * dim + d, mask=row < tokens, other=0.0)
    c = tl.load(cos_ptr + col * d, mask=row < tokens, other=0.0)
    s = tl.load(sin_ptr + col * d, mask=row < tokens, other=0.0)
    
    # Perform the rotation
    x1_rot = x1 * c - x2 * s
    x2_rot = x2 * c + x1 * s
    
    # Store the results back to the output
    tl.store(out_ptr + row * dim + col * heads * dim, x1_rot, mask=row < tokens)
    tl.store(out_ptr + row * dim + col * heads * dim + d, x2_rot, mask=row < tokens)

def kernel(x, cos, sin):
    # Get the dimensions
    tokens, heads, dim = x.shape
    assert dim % 2 == 0, "Dimension must be even"
    
    # Allocate the output tensor
    out = torch.empty_like(x)
    
    # Define the grid and block sizes
    BLOCK_SIZE = 128
    grid = (triton.cdiv(tokens, BLOCK_SIZE), heads)
    
    # Launch the kernel
    _rope_kernel[grid](x, cos, sin, out, tokens, heads, dim, BLOCK_SIZE=BLOCK_SIZE)
    
    return out
