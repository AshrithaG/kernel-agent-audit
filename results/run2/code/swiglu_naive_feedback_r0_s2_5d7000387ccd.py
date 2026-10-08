import torch
import triton
import triton.language as tl

@triton.jit
def swiglu_kernel(x_ptr, out_ptr, N, H, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    
    # Calculate the row and column indices
    row_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_idx = bid
    
    # Ensure we don't go out of bounds
    mask = row_idx < N
    
    # Load the input values
    x_first_half = tl.load(x_ptr + col_idx * N + row_idx, mask=mask, other=0.0)
    x_second_half = tl.load(x_ptr + col_idx * N + row_idx + H, mask=mask, other=0.0)
    
    # Apply SiLU activation to the first half
    x_first_half_silu = x_first_half * tl.sigmoid(x_first_half)
    
    # Multiply the first half by the second half
    out = x_first_half_silu * x_second_half
    
    # Store the result
    tl.store(out_ptr + col_idx * N + row_idx, out, mask=mask)

def kernel(x):
    N, H = x.shape
    assert H % 2 == 0, "The last dimension must be even."
    H //= 2
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Define grid and block dimensions
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE), H)
    
    # Launch the kernel
    swiglu_kernel[grid](x.contiguous(), out, N, H, BLOCK_SIZE)
    
    return out

# Example usage
if __name__ == "__main__":
    import torch.nn.functional as F
    import math
    
    # Create a random input tensor
    x = torch.randn((1024, 512), device='cuda', dtype=torch.float16)
    
    # Reference implementation
    ref_out = (F.silu(x[..., :256].float()) * x[..., 256:].float()).to(x.dtype)
    
    # Triton implementation
    triton_out = kernel(x)
    
    # Check if the outputs are close
    assert torch.allclose(ref_out, triton_out, atol=1e-3, rtol=1e-3), "Outputs do not match!"
    
    print("Reference output:", ref_out)
    print("Triton output:", triton_out)
