import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, B, H, D, BLOCK_SIZE: tl.constexpr):
    # Each thread handles one token
    token_idx = tl.program_id(0)
    if token_idx >= B:
        return
    
    # Load cos and sin values
    cos = tl.load(cos_ptr + token_idx * (D // 2), mask=token_idx < B, other=0.0)
    sin = tl.load(sin_ptr + token_idx * (D // 2), mask=token_idx < B, other=0.0)
    
    # Load x values
    x1 = tl.load(x_ptr + token_idx * H * D, mask=token_idx < B, other=0.0)
    x2 = tl.load(x_ptr + token_idx * H * D + (D // 2), mask=token_idx < B, other=0.0)
    
    # Compute the rotated values
    x1_rotated = x1 * cos - x2 * sin
    x2_rotated = x2 * cos + x1 * sin
    
    # Store the results back to out_ptr
    tl.store(out_ptr + token_idx * H * D, x1_rotated, mask=token_idx < B)
    tl.store(out_ptr + token_idx * H * D + (D // 2), x2_rotated, mask=token_idx < B)

def kernel(x, cos, sin):
    B, H, D = x.shape
    assert D % 2 == 0, "Dimension must be even"
    out = torch.empty_like(x)
    
    # Launch the kernel
    grid = lambda META: (triton.cdiv(B, META['BLOCK_SIZE']),)
    rope_kernel[grid](x, cos, sin, out, B, H, D, BLOCK_SIZE=1024)
    
    return out

# Example usage:
# x = torch.randn(1024, 32, 128, device='cuda', dtype=torch.float16)
# cos = torch.randn(1024, 64, device='cuda', dtype=torch.float16)
# sin = torch.randn(1024, 64, device='cuda', dtype=torch.float16)
# result = kernel(x, cos, sin)
