import torch
import triton
import triton.language as tl

@triton.jit
def _add_rmsnorm_kernel(x_ptr, r_ptr, w_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # Calculate the index of the current program instance
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Load the data into registers, masking out-of-bounds accesses
    mask = idx < n_elements
    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr, mask=mask, other=0.0).to(tl.float32)
    
    # Compute the sum of squares
    sum_squares = tl.sum(x * x, axis=0)
    # Compute the RMSNorm factor
    rmsnorm_factor = tl.rsqrt(sum_squares + 1e-6)
    # Compute the output
    out = (x + r) * rmsnorm_factor * w
    
    # Store the result back to global memory
    tl.store(out_ptr + idx, out.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, r, w):
    x = x.contiguous()
    r = r.contiguous()
    w = w.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(BLOCK)
    _add_rmsnorm_kernel[(n_elements // BLOCK_SIZE,)](x, r, w, out, n_elements, BLOCK=BLOCK_SIZE)
    return out
