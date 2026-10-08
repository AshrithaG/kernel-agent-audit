import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh

@triton.jit
def _rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, dim, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < dim

    # Load inputs
    x1 = tl.load(x_ptr + token * dim + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + token * dim + cols + dim, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute rotated embeddings
    x1_rot = x1 * c - x2 * s
    x2_rot = x2 * c + x1 * s

    # Store results
    tl.store(out_ptr + token * dim + cols, x1_rot.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(out_ptr + token * dim + cols + dim, x2_rot.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, cos, sin):
    x = x.contiguous()
    out = torch.empty_like(x)
    tokens, heads, dim = x.shape
    dim_half = dim // 2

    # Launch kernel
    grid = (tokens, heads)
    _rope_kernel[grid](x, cos, sin, out, dim_half, BLOCK=triton.next_power_of_2(dim_half))

    return out
