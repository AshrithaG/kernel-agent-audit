import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, B, H, D, BLOCK_SIZE: tl.constexpr):
    # Each thread handles one head for a given token
    token_idx = tl.program_id(0) // H
    head_idx = tl.program_id(0) % H
    
    if token_idx >= B:
        return
    
    # Load cos and sin values
    cos = tl.load(cos_ptr + token_idx * (D // 2))
    sin = tl.load(sin_ptr + token_idx * (D // 2))
    
    # Load x values
    x1_offset = token_idx * H * D + head_idx * D
    x2_offset = x1_offset + (D // 2)
    x1 = tl.load(x_ptr + x1_offset, mask=token_idx < B, other=0.0)
    x2 = tl.load(x_ptr + x2_offset, mask=token_idx < B, other=0.0)
    
    # Compute the rotated values
    x1_rotated = x1 * cos - x2 * sin
    x2_rotated = x2 * cos + x1 * sin
    
    # Store the results back to out_ptr
    out1_offset = x1_offset
    out2_offset = x2_offset
    tl.store(out_ptr + out1_offset, x1_rotated, mask=token_idx < B)
    tl.store(out_ptr + out2_offset, x2_rotated, mask=token_idx < B)

def kernel(x, cos, sin):
    B, H, D = x.shape
    assert D % 2 == 0, "Dimension must be even"
    out = torch.empty_like(x)
    
    grid = lambda META: (triton.cdiv(B * H, META['BLOCK_SIZE']),)
    rope_kernel[grid](x, cos, sin, out, B, H, D, BLOCK_SIZE=1024)
    
    return out

# Example usage:
# x = torch.randn(1024, 32, 128, device='cuda', dtype=torch.float16)
# cos = torch.randn(1024, 64, device='cuda', dtype=torch.float16)
# sin = torch.randn(1024, 64, device='cuda', dtype=torch.float16)
# result = kernel(x, cos, sin)
