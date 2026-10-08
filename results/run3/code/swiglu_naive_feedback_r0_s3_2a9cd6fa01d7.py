import torch
import triton
import triton.language as tl

# Define the Triton kernel for SwiGLU
@triton.jit
def _swiglu_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # Get the current program ID (block index)
    pid = tl.program_id(0)
    # Generate indices for the current block
    idx = tl.arange(0, BLOCK)
    # Calculate the start index for the current block
    start_idx = pid * BLOCK
    # Create a mask to ensure we do not access out-of-bounds elements
    mask = idx < n_elements
    
    # Load the input elements
    x = tl.load(x_ptr + start_idx + idx, mask=mask, other=0.0).to(tl.float32)
    
    # Split the elements into two halves
    n_half = n_elements // 2
    first_half = x[:n_half]
    second_half = x[n_half:]
    
    # Apply SiLU activation to the first half
    sielu = first_half * tl.sigmoid(first_half)
    
    # Multiply the SiLU result with the second half
    result = sielu * second_half
    
    # Store the result back to the output buffer
    tl.store(out_ptr + start_idx + idx, result.to(out_ptr.dtype.element_ty), mask=mask)

# Define the Python function to launch the Triton kernel
def kernel(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.shape[-1]
    grid = (triton.cdiv(n_elements, triton.next_power_of_2(n_elements)),)
    _swiglu_kernel[grid](x, out, n_elements, BLOCK=triton.next_power_of_2(n_elements))
    return out
