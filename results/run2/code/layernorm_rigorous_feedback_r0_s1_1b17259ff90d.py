import torch
import triton
import triton.language as tl
import math

@triton.jit
def layernorm_kernel(x_ptr, w_ptr, b_ptr, out_ptr, n_elements, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    indices = block_start + tl.arange(0, BLOCK_SIZE)
    mask = indices < n_elements

    x = tl.load(x_ptr + indices, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_elements
    mean_sqr = tl.sum(x**2, axis=0) / n_elements
    var = mean_sqr - mean**2 + eps
    inv_var = 1.0 / tl.sqrt(var)
    normalized_x = (x - mean) * inv_var

    w = tl.load(w_ptr, mask=mask, other=0.0)
    b = tl.load(b_ptr, mask=mask, other=0.0)
    out = normalized_x * w + b

    tl.store(out_ptr + indices, out, mask=mask)

def kernel(x, w, b):
    assert x.is_cuda and w.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == torch.float16 and w.dtype == torch.float16 and b.dtype == torch.float16, "Inputs must be float16 tensors"
    assert x.ndim == 2 and w.ndim == 1 and b.ndim == 1, "Inputs must have the correct shapes"
    assert x.shape[1] == w.shape[0] == b.shape[0], "Last dimension of x must match the shape of w and b"

    n_elements = x.shape[1]
    BLOCK_SIZE = 128
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)

    out = torch.empty_like(x)

    layernorm_kernel[grid_size, BLOCK_SIZE](
        x_ptr=x.data_ptr(),
        w_ptr=w.data_ptr(),
        b_ptr=b.data_ptr(),
        out_ptr=out.data_ptr(),
        n_elements=n_elements,
        eps=1e-5,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out

# Example usage:
# x = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
# w = torch.randn(512, device='cuda', dtype=torch.float16)
# b = torch.randn(512, device='cuda', dtype=torch.float16)
# result = kernel(x, w, b)
