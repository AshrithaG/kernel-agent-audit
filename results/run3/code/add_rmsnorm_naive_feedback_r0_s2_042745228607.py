import torch
import triton
import triton.language as tl

@triton.jit
def _add_rmsnorm(x_ptr, r_ptr, w_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # Get the index of the current program instance
    idx = tl.program_id(0)
    # Create a range of indices for the block
    cols = tl.arange(0, BLOCK)
    # Mask to ensure we only access valid elements
    mask = cols < n_elements
    # Load the input data, defaulting to 0.0 if out of bounds
    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    
    # Compute the sum of squares
    h = x + r
    sum_squares = tl.sum(h * h, axis=0)
    # Compute the RMSNorm
    rms = tl.sqrt(sum_squares / n_elements + 1e-6)
    # Apply the weight and store the result
    tl.store(out_ptr + idx, (h / rms * w).to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, r, w):
    # Ensure the inputs are contiguous
    x = x.contiguous()
    r = r.contiguous()
    w = w.contiguous()
    # Create an output tensor of the same shape as x
    out = torch.empty_like(x)
    # Get the number of elements in the last dimension
    n_elements = x.shape[-1]
    # Launch the Triton kernel
    grid = (x.numel(),)
    _add_rmsnorm[grid](x, r, w, out, n_elements, BLOCK=triton.next_power_of_2(n_elements))
    return out
