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
    
    # Split the input into two halves
    half_size = n_elements // 2
    first_half = input_val[:half_size]
    second_half = input_val[half_size:]
    
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
    
    # Split the workload into smaller chunks to avoid exceeding Triton's maximum tensor numel
    MAX_NUMEL = 1048576
    num_chunks = (n_elements + MAX_NUMEL - 1) // MAX_NUMEL
    
    for chunk in range(num_chunks):
        start_idx = chunk * MAX_NUMEL
        end_idx = min((chunk + 1) * MAX_NUMEL, n_elements)
        chunk_size = end_idx - start_idx
        
        BLOCK = triton.next_power_of_2(chunk_size)
        grid = (triton.cdiv(chunk_size, BLOCK),)
        
        _swiglu_kernel[grid](x + start_idx, out + start_idx, chunk_size, BLOCK=BLOCK)
    
    return out
