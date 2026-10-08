import torch
import triton
import triton.language as tl

@triton.jit
def gelu_tanh_approx_kernel(
    x_ptr, b_ptr, out_ptr,
    N: tl.constexpr, M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the row index and column index for the current thread
    pid = tl.program_id(0)
    block_pid = tl.program_id(1)
    
    row = pid
    col = block_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load the bias value for the current row
    b_val = tl.load(b_ptr + row)
    
    # Load the input values for the current row and columns
    x_vals = tl.load(x_ptr + row * M + col, mask=col < M, other=0.0)
    
    # Add the bias to the input values
    x_vals += b_val
    
    # Approximate GELU using tanh
    cdf = 0.5 * (1.0 + tl.tanh((x_vals / 1.41421356237) * (1.0 + 0.044715 * x_vals * x_vals)))
    out_vals = x_vals * cdf
    
    # Store the results back to the output pointer
    tl.store(out_ptr + row * M + col, out_vals, mask=col < M)

def kernel(x, b):
    # Ensure the inputs are on the same device
    if x.device != b.device:
        raise ValueError("Inputs must be on the same device")
    
    # Get the shape of the input tensor
    N, M = x.shape
    
    # Create an output tensor with the same shape as the input tensor
    out = torch.empty_like(x)
    
    # Define the grid and block sizes for the Triton kernel
    BLOCK_SIZE = 128
    grid = lambda META: (N, triton.cdiv(M, META['BLOCK_SIZE']))
    
    # Launch the Triton kernel
    gelu_tanh_approx_kernel[grid](x, b, out, N, M, BLOCK_SIZE)
    
    return out

# Example usage
if __name__ == "__main__":
    import torch.nn.functional as F
    
    # Create example input tensors
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    b = torch.randn(1024, device='cuda', dtype=torch.float16)
    
    # Run the reference implementation
    ref_output = F.gelu(x.float() + b.float(), approximate="tanh").to(x.dtype)
    
    # Run the Triton kernel implementation
    triton_output = kernel(x, b)
    
    # Check if the outputs are the same within floating-point tolerance
    assert torch.allclose(ref_output, triton_output, atol=1e-3, rtol=1e-3)
    
    print("Outputs match within floating-point tolerance.")
