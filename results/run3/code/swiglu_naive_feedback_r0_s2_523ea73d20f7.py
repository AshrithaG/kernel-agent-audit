import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def _swiglu_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_elements
    input_data = tl.load(input_ptr + pid * n_elements + cols, mask=mask, other=0.0).to(tl.float32)
    
    n_half = n_elements // 2
    first_half = input_data[:n_half]
    second_half = input_data[n_half:]
    
    # Apply SiLU activation to the first half
    sigmoid_part = libdevice.sigmoid(first_half)
    silu_result = first_half * sigmoid_part
    
    # Element-wise multiplication
    result = silu_result * second_half
    
    tl.store(output_ptr + pid * n_elements + cols, result.to(output_ptr.dtype.element_ty), mask=mask)

def kernel(input_tensor):
    input_tensor = input_tensor.contiguous()
    output_tensor = torch.empty_like(input_tensor)
    batch_size, n_elements = input_tensor.shape
    _swiglu_kernel[(batch_size,)](input_tensor, output_tensor, n_elements, BLOCK=triton.next_power_of_2(n_elements))
    return output_tensor
