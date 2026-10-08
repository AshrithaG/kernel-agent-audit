import torch
import triton
import triton.language as tl

@triton.jit
def _rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, dim, num_tokens, num_heads, BLOCK: tl.constexpr):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Calculate the starting index for this block
    start_idx = token_idx * num_heads * dim + head_idx * dim
    
    # Load x1 and x2
    cols = tl.arange(0, BLOCK)
    mask = cols < dim // 2
    x1 = tl.load(x_ptr + start_idx + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + start_idx + cols + dim // 2, mask=mask, other=0.0).to(tl.float32)
    
    # Load cos and sin
    cos = tl.load(cos_ptr + token_idx * (dim // 2) + cols, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + token_idx * (dim // 2) + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Perform the rotation
    x1_rotated = x1 * cos - x2 * sin
    x2_rotated = x2 * cos + x1 * sin
    
    # Store the results
    tl.store(out_ptr + start_idx + cols, x1_rotated.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(out_ptr + start_idx + cols + dim // 2, x2_rotated.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, cos, sin):
    x = x.contiguous()
    out = torch.empty_like(x)
    num_tokens, num_heads, dim = x.shape
    BLOCK_SIZE = 32  # Choose a block size that fits well with the architecture
    _rope_kernel[(num_tokens, num_heads), (1, BLOCK_SIZE)](
        x_ptr=x.data_ptr(),
        cos_ptr=cos.data_ptr(),
        sin_ptr=sin.data_ptr(),
        out_ptr=out.data_ptr(),
        dim=dim,
        num_tokens=num_tokens,
        num_heads=num_heads,
        BLOCK=BLOCK_SIZE
    )
    return out
