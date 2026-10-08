import torch
import triton
import triton.language as tl

@triton.jit
def _add_rmsnorm_kernel(x_ptr, r_ptr, w_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr, other=0.0).to(tl.float32)
    
    h = x + r
    sum_squares = tl.sum(h * h, axis=0)
    rms = tl.rsqrt(sum_squares + 1e-6)
    normed_h = h * rms
    
    tl.store(out_ptr + row * n_cols + cols, (normed_h * w).to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, r, w):
    x = x.contiguous()
    r = r.contiguous()
    w = w.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    _add_rmsnorm_kernel[(n_rows,)](x, r, w, out, n_cols, BLOCK=triton.next_power_of_2(n_cols))
    return out
