import torch
import triton
import triton.language as tl
from triton.language import extra

@triton.jit
def _swiglu_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    # Calculate the row and column indices
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_elements
    
    # Load the input data
    input_data = tl.load(input_ptr + row * n_elements + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Split the input into two halves
    n_half = n_elements // 2
    first_half = input_data[:n_half]
    second_half = input_data[n_half:]
    
    # Apply SiLU to the first half
    first_half_silu = first_half * extra.triton.sigmoid(first_half)
    
    # Multiply the SiLU result with the second half
    result = first_half_silu * second_half
    
    # Store the result back to the output
    tl.store(output_ptr + row * n_elements + cols, result.to(output_ptr.dtype.element_ty), mask=mask)

def kernel(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_elements = x.shape
    _swiglu_kernel[(n_rows,)](x, out, n_elements, BLOCK=triton.next_power_of_2(n_elements))
    return out
