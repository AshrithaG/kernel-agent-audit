import torch
import triton
import triton.language as tl

@triton.jit
def _rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, n_tokens, n_heads, d, BLOCK: tl.constexpr):
    head = tl.program_id(0)
    token = tl.program_id(1)
    
    # Calculate the indices for the current block
    cols = tl.arange(0, BLOCK)
    mask = cols < d
    
    # Load x1 and x2
    x1_ptr = x_ptr + token * n_heads * d * 2 + head * d * 2
    x2_ptr = x_ptr + token * n_heads * d * 2 + head * d * 2 + d
    x1 = tl.load(x1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Load cos and sin
    cos_ptr = cos_ptr + token * d
    sin_ptr = sin_ptr + token * d
    c = tl.load(cos_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute the rotated values
    x1_rotated = x1 * c - x2 * s
    x2_rotated = x2 * c + x1 * s
    
    # Store the results
    out1_ptr = out_ptr + token * n_heads * d * 2 + head * d * 2
    out2_ptr = out_ptr + token * n_heads * d * 2 + head * d * 2 + d
    tl.store(out1_ptr + cols, x1_rotated.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(out2_ptr + cols, x2_rotated.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, cos, sin):
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    out = torch.empty_like(x)
    n_tokens, n_heads, d = x.shape
    d //= 2
    BLOCK = triton.next_power_of_2(d)
    _rope_kernel[(n_heads, n_tokens)](x, cos, sin, out, n_tokens, n_heads, d, BLOCK=BLOCK)
    return out
