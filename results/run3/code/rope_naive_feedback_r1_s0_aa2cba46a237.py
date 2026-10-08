import torch
import triton
import triton.language as tl

@triton.jit
def _rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, n_tokens, n_heads, n_dim, BLOCK: tl.constexpr):
    head = tl.program_id(0)
    token = tl.program_id(1)
    
    d = n_dim // 2
    cols = tl.arange(0, BLOCK)
    mask = cols < d
    
    # Load x1 and x2
    x1_ptr = x_ptr + (token * n_heads + head) * n_dim
    x1 = tl.load(x1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    x2_ptr = x1_ptr + d
    x2 = tl.load(x2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Load cos and sin
    cos_ptr = cos_ptr + token * d
    cos = tl.load(cos_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    sin_ptr = sin_ptr + token * d
    sin = tl.load(sin_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute the rotated values
    x1_rotated = x1 * cos - x2 * sin
    x2_rotated = x2 * cos + x1 * sin
    
    # Store the results
    out1_ptr = out_ptr + (token * n_heads + head) * n_dim
    tl.store(out1_ptr + cols, x1_rotated.to(out_ptr.dtype.element_ty), mask=mask)
    
    out2_ptr = out1_ptr + d
    tl.store(out2_ptr + cols, x2_rotated.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, cos, sin):
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    out = torch.empty_like(x)
    
    n_tokens, n_heads, n_dim = x.shape
    BLOCK = triton.next_power_of_2(n_dim // 2)
    
    grid = (n_heads, n_tokens)
    _rope_kernel[grid](x, cos, sin, out, n_tokens, n_heads, n_dim, BLOCK=BLOCK)
    return out
