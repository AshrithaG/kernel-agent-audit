import torch
import triton
import triton.language as tl
import math

# Triton kernel for rotary position embedding
@triton.jit
def rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, tokens, heads, dim, BLOCK_SIZE: tl.constexpr):
    # Get the position of the current thread within the block
    row = tl.program_id(0)  # token index
    col = tl.program_id(1)  # head index
    # Calculate the base pointers for the current token and head
    x_base = row * heads * dim + col * dim
    cos_base = row * (dim // 2)
    sin_base = row * (dim // 2)
    out_base = row * heads * dim + col * dim
    # Load x, cos, and sin into shared memory
    d = dim // 2
    x1 = tl.load(x_ptr + x_base, mask=row < tokens, other=0.0).to(tl.float16)
    x2 = tl.load(x_ptr + x_base + d, mask=row < tokens, other=0.0).to(tl.float16)
    c = tl.load(cos_ptr + cos_base, mask=row < tokens, other=0.0).to(tl.float16)
    s = tl.load(sin_ptr + sin_base, mask=row < tokens, other=0.0).to(tl.float16)
    # Apply the rotation
    x1_rot = x1 * c - x2 * s
    x2_rot = x2 * c + x1 * s
    # Store the results back to global memory
    tl.store(out_ptr + out_base, x1_rot, mask=row < tokens)
    tl.store(out_ptr + out_base + d, x2_rot, mask=row < tokens)

# Triton wrapper function
def kernel(x, cos, sin):
    tokens, heads, dim = x.shape
    assert dim % 2 == 0, "Dimension must be even"
    out = torch.empty_like(x)
    # Define the grid and block dimensions
    BLOCK_SIZE = 128
    grid = (tokens, heads)
    # Launch the Triton kernel
    rope_kernel[grid](x.contiguous(), cos.contiguous(), sin.contiguous(), out.contiguous(), tokens, heads, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out

# Example usage
if __name__ == "__main__":
    tokens = 128
    heads = 8
    dim = 64
    x = torch.randn(tokens, heads, dim, device='cuda', dtype=torch.float16)
    cos = torch.randn(tokens, dim // 2, device='cuda', dtype=torch.float16)
    sin = torch.randn(tokens, dim // 2, device='cuda', dtype=torch.float16)
    out = kernel(x, cos, sin)
    print(out.shape)
