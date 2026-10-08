import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import erf

@triton.jit
def _swiglu_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    # Get the current program instance ID
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    
    # Load the input values
    input_val = tl.load(input_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    
    # Determine the position of the element in the tensor
    row_idx = idx // (n_elements // 2)
    col_idx = idx % (n_elements // 2)
    
    # Split the input into two halves
    if col_idx < (n_elements // 2) // 2:
        # First half
        first_half = input_val
        second_half = tl.load(input_ptr + row_idx * (n_elements // 2) + col_idx + (n_elements // 2) // 2, mask=mask, other=0.0).to(tl.float32)
        
        # Apply SiLU to the first half
        silu_val = first_half * tl.sigmoid(first_half)
        
        # Multiply the SiLU result with the second half
        result = silu_val * second_half
    else:
        # Second half
        result = 0.0
    
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
