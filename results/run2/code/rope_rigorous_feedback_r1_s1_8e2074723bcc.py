import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, tokens, heads, dim, BLOCK_SIZE: tl.constexpr):
    # Get the thread ID
    idx = tl.program_id(0) * BLOCK_SIZE + tl.thread_idx(0)
    # Check if the thread is within bounds
    if idx >= tokens * heads:
        return
    
    # Calculate the row and column indices
    row = idx // heads
    col = idx % heads
    
    # Load x, cos, and sin into registers
    d = dim // 2
    x1 = tl.load(x_ptr + idx * dim, mask=idx < tokens * heads, other=0.0)
    x2 = tl.load(x_ptr + idx * dim + d, mask=idx < tokens * heads, other=0.0)
    c = tl.load(cos_ptr + row * d, mask=row < tokens, other=0.0)
    s = tl.load(sin_ptr + row * d, mask=row < tokens, other=0.0)
    
    # Apply the rotation
    out1 = x1 * c - x2 * s
    out2 = x2 * c + x1 * s
    
    # Store the result back to global memory
    tl.store(out_ptr + idx * dim, out1, mask=idx < tokens * heads)
    tl.store(out_ptr + idx * dim + d, out2, mask=idx < tokens * heads)

def kernel(x, cos, sin):
    tokens, heads, dim = x.shape
    assert dim % 2 == 0, "Dimension must be even"
    assert cos.shape == (tokens, dim // 2)
    assert sin.shape == (tokens, dim // 2)
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Define the grid and block size
    BLOCK_SIZE = 128
    grid_size = (tokens * heads + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    rope_kernel[grid_size, BLOCK_SIZE](
        x.data_ptr(),
        cos.data_ptr(),
        sin.data_ptr(),
        out.data_ptr(),
        tokens,
        heads,
        dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Example usage:
# x = torch.randn(1024, 32, 128, device='cuda', dtype=torch.float16)
# cos = torch.randn(1024, 64, device='cuda', dtype=torch.float16)
# sin = torch.randn(1024, 64, device='cuda', dtype=torch.float16)
# out = kernel(x, cos, sin)
