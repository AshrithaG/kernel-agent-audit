import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import erf

@triton.jit
def _swiglu_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    # Get the current program instance ID
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    
    # Calculate the index for the first half and second half
    half_size = n_elements // 2
    first_half_idx = idx[:half_size]
    second_half_idx = idx[half_size:] - half_size
    
    # Load the input values for the first half and second half
    first_half = tl.load(input_ptr + first_half_idx, mask=mask[:half_size], other=0.0).to(tl.float32)
    second_half = tl.load(input_ptr + half_size + second_half_idx, mask=mask[half_size:], other=0.0).to(tl.float32)
    
    # Apply SiLU to the first half
    silu_val = first_half * tl.sigmoid(first_half)
    
    # Multiply the SiLU result with the second half
    result = silu_val * second_half
    
    # Store the result back to the output pointer
    tl.store(output_ptr + idx, result.to(output_ptr.dtype.element_ty), mask=mask)

def kernel(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK = 1024  # Set a reasonable block size
    grid = (triton.cdiv(n_elements, BLOCK),)
    _swiglu_kernel[grid](x, out, n_elements, BLOCK=BLOCK)
    return out
