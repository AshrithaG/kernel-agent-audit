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
    x1_rotated = x1 * c - x2 * s
    x2_rotated = x2 * c + x1 * s
    
    # Store the results back to the output pointer
    tl.store(out_ptr + row * dim + col * heads * dim, x1_rotated, mask=row < tokens)
    tl.store(out_ptr + row * dim + col * heads * dim + d, x2_rotated, mask=row < tokens)

def kernel(x, cos, sin):
    # Ensure the input shapes are compatible
    assert x.shape[0] == cos.shape[0]
    assert x.shape[-1] == cos.shape[-1] * 2
    assert x.dtype == torch.float16
    assert cos.dtype == torch.float16
    assert sin.dtype == torch.float16

    # Get the dimensions
    tokens, heads, dim = x.shape
    d = dim // 2

    # Allocate the output tensor
    out = torch.empty_like(x)

    # Define the grid and block sizes
    BLOCK_SIZE = 128
    num_blocks = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    grid = lambda META: (num_blocks, heads)
    _rope_kernel[grid](x, cos, sin, out, tokens, heads, dim, BLOCK_SIZE=BLOCK_SIZE)

    return out

# Example usage
if __name__ == "__main__":
    # Create some dummy data
    tokens = 1024
    heads = 16
    dim = 64

    x = torch.randn(tokens, heads, dim, dtype=torch.float16, device='cuda')
    cos = torch.randn(tokens, dim // 2, dtype=torch.float16, device='cuda')
    sin = torch.randn(tokens, dim // 2, dtype=torch.float16, device='cuda')

    # Run the reference implementation
    ref_out = rope(x, cos, sin)

    # Run the Triton implementation
    triton_out = kernel(x, cos, sin)

    # Check if the outputs are close
    print(torch.allclose(ref_out, triton_out))
