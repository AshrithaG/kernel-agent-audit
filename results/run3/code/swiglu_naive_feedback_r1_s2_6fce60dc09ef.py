import torch
import triton
import triton.language as tl

@triton.jit
def _swiglu(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # Calculate the index of the current block
    idx = tl.program_id(0) * BLOCK
    # Create a range of indices for the current block
    offsets = tl.arange(0, BLOCK)
    # Calculate the indices for the first and second halves
    half_size = n_elements // 2
    first_half_indices = idx + offsets[:half_size]
    second_half_indices = idx + offsets[half_size:2*half_size]
    
    # Load the first and second halves
    first_half = tl.load(x_ptr + first_half_indices, mask=first_half_indices < half_size, other=0.0).to(tl.float32)
    second_half = tl.load(x_ptr + second_half_indices, mask=second_half_indices < half_size, other=0.0).to(tl.float32)
    
    # Apply SiLU to the first half
    first_half_silu = first_half * tl.sigmoid(first_half)
    
    # Multiply the SiLU result with the second half
    result = first_half_silu * second_half
    
    # Store the result back into the output buffer
    tl.store(out_ptr + idx + offsets, result.to(out_ptr.dtype.element_ty), mask=offsets < n_elements)

def kernel(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.shape[-1]
    BLOCK = triton.next_power_of_2(n_elements)
    grid = (x.numel() // BLOCK,)
    _swiglu[grid](x, out, n_elements, BLOCK=BLOCK)
    return out
