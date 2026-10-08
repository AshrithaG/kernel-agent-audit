import torch
import triton
import triton.language as tl

@triton.jit
def _swiglu_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # Calculate the index of the current program instance
    idx = tl.program_id(0)
    # Calculate the range of elements to process
    cols = tl.arange(0, BLOCK)
    # Mask to ensure we don't access out of bounds
    mask = cols < n_elements
    # Load the first half of the elements
    x_first_half = tl.load(x_ptr + idx * n_elements + cols, mask=mask, other=0.0).to(tl.float32)
    # Load the second half of the elements
    x_second_half = tl.load(x_ptr + idx * n_elements + cols + n_elements // 2, mask=mask, other=0.0).to(tl.float32)
    # Apply the SiLU activation to the first half
    silu_result = x_first_half * tl.sigmoid(x_first_half)
    # Multiply the SiLU result with the second half
    result = silu_result * x_second_half
    # Store the result back to the output pointer
    tl.store(out_ptr + idx * n_elements + cols, result.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x):
    # Ensure the input tensor is contiguous
    x = x.contiguous()
    # Create an output tensor with the same shape and type as the input
    out = torch.empty_like(x)
    # Get the number of elements in the last dimension
    n_elements = x.shape[-1]
    # Calculate the grid size based on the number of rows
    grid = (x.numel() // n_elements,)
    # Launch the Triton kernel
    _swiglu_kernel[grid](x, out, n_elements, BLOCK=triton.next_power_of_2(n_elements))
    return out
