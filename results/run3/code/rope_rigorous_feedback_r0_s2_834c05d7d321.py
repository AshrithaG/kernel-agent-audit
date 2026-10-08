import torch
import triton
import triton.language as tl

@triton.jit
def _rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, tokens, heads, dim, BLOCK: tl.constexpr):
    # Get the current token index
    token = tl.program_id(0)
    # Get the current head index
    head = tl.program_id(1)
    # Calculate the offset for this token and head
    offset = token * heads * dim + head * dim
    
    # Load the input tensor
    x = tl.load(x_ptr + offset, mask=(token < tokens) & (head < heads), other=0.0).to(tl.float32)
    
    # Split the input into two halves
    d = dim // 2
    x1 = x[:d]
    x2 = x[d:]
    
    # Load the cosine and sine values
    cos = tl.load(cos_ptr + token * d, mask=(token < tokens), other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + token * d, mask=(token < tokens), other=0.0).to(tl.float32)
    
    # Perform the rotation
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x2 * cos + x1 * sin
    
    # Concatenate the results
    rotated_x = tl.cat((rotated_x1, rotated_x2))
    
    # Store the result
    tl.store(out_ptr + offset, rotated_x.to(out_ptr.dtype.element_ty), mask=(token < tokens) & (head < heads))

def kernel(x, cos, sin):
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    out = torch.empty_like(x)
    tokens, heads, dim = x.shape
    BLOCK = 128  # Choose an appropriate block size
    grid = (tokens, heads)
    _rope_kernel[grid](x, cos, sin, out, tokens, heads, dim, BLOCK=BLOCK)
    return out
