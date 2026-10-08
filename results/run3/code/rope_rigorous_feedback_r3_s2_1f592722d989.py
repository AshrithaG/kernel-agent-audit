import torch
import triton
import triton.language as tl
from triton.language import extra

# Define the block size
BLOCK_SIZE = 128

@triton.jit
def _rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, n_tokens, n_heads, n_dim, BLOCK: tl.constexpr):
    # Define shared memory
    smem_x1 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    smem_x2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    smem_c = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    smem_s = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    head = tl.program_id(0)
    token = tl.program_id(1)
    d = n_dim // 2
    cols = tl.arange(0, BLOCK)
    mask = cols < d
    
    # Load x1 and x2 into shared memory
    x1_ptr = x_ptr + token * n_heads * n_dim + head * n_dim
    x2_ptr = x1_ptr + d
    x1 = tl.load(x1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    smem_x1[cols] = x1
    smem_x2[cols] = x2
    
    # Load cos and sin into shared memory
    cos_ptr = cos_ptr + token * d
    sin_ptr = sin_ptr + token * d
    c = tl.load(cos_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    smem_c[cols] = c
    smem_s[cols] = s
    
    # Compute the rotated values
    x1_rotated = smem_x1[cols] * smem_c[cols] - smem_x2[cols] * smem_s[cols]
    x2_rotated = smem_x2[cols] * smem_c[cols] + smem_x1[cols] * smem_s[cols]
    
    # Store the results
    tl.store(out_ptr + token * n_heads * n_dim + head * n_dim + cols, x1_rotated.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(out_ptr + token * n_heads * n_dim + head * n_dim + d + cols, x2_rotated.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, cos, sin):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_tokens, n_heads, n_dim = x.shape
    _rope_kernel[(n_heads, n_tokens)](x, cos, sin, out, n_tokens, n_heads, n_dim, BLOCK=BLOCK_SIZE)
    return out
