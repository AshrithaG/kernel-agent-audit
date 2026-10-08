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
    cos = tl.load(cos_ptr + token_idx * (D // 2))
    sin = tl.load(sin_ptr + token_idx * (D // 2))
    
    # Load x values
    for head_idx in range(H):
        x1 = tl.load(x_ptr + token_idx * H * D + head_idx * D, mask=token_idx < B, other=0.0)
        x2 = tl.load(x_ptr + token_idx * H * D + head_idx * D + (D // 2), mask=token_idx < B, other=0.0)
        
        # Compute the rotated values
        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x2 * cos + x1 * sin
        
        # Store the results back to out_ptr
        tl.store(out_ptr + token_idx * H * D + head_idx * D, x1_rotated, mask=token_idx < B)
        tl.store(out_ptr + token_idx * H * D + head_idx * D + (D // 2), x2_rotated, mask=token_idx < B)

def kernel(x, cos, sin):
    B, H, D = x.shape
    assert D % 2 == 0, "Dimension D must be even"
    assert cos.shape == (B, D // 2) and sin.shape == (B, D // 2), "cos and sin must have shape (B, D // 2)"
    
    out = torch.empty_like(x)
    
    grid = lambda META: (triton.cdiv(B, META['BLOCK_SIZE']),)
    rope_kernel[grid](x, cos, sin, out, B, H, D, BLOCK_SIZE=1024)
    
    return out

# Example usage
if __name__ == "__main__":
    import math
    torch.manual_seed(0)
    B, H, D = 32, 8, 64
    x = torch.rand((B, H, D), device='cuda', dtype=torch.float16)
    cos = torch.cos(torch.arange(D // 2, device='cuda', dtype=torch.float16) * math.pi / D)
    sin = torch.sin(torch.arange(D // 2, device='cuda', dtype=torch.float16) * math.pi / D)
    
    out = kernel(x, cos, sin)
    print(out)
