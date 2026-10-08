import torch
import triton
import triton.language as tl
import math

# Triton kernel for RMSNorm
@triton.jit
def rmsnorm_kernel(
    x_ptr, w_ptr, out_ptr,
    stride_x, stride_w, stride_out,
    N, M,
    BLOCK_SIZE: tl.constexpr
):
    # Define the program ID
    pid = tl.program_id(axis=0)
    # Calculate the row index
    row_idx = pid * BLOCK_SIZE
    # Load the row from x and w
    x_row = tl.load(x_ptr + row_idx * stride_x, mask=row_idx < N, other=0.0)
    w_row = tl.load(w_ptr + row_idx * stride_w, mask=row_idx < N, other=0.0)
    # Compute the squared sum
    sq_sum = tl.sum(x_row ** 2, axis=1)
    # Add epsilon to avoid division by zero
    sq_sum += 1e-6
    # Compute the reciprocal square root
    rsqrt = 1.0 / tl.sqrt(sq_sum)
    # Normalize and scale by weight
    out_row = x_row * rsqrt * w_row
    # Store the result
    tl.store(out_ptr + row_idx * stride_out, out_row, mask=row_idx < N)

# Triton RMSNorm function
def triton_rmsnorm(x, w):
    # Get the shape of the input tensors
    N, M = x.shape
    # Allocate output tensor
    out = torch.empty_like(x)
    # Define the grid size
    grid_size = (triton.cdiv(N, 32),)
    # Launch the kernel
    rmsnorm_kernel[grid_size](
        x.contiguous(), w.contiguous(), out.contiguous(),
        x.stride(0), w.stride(0), out.stride(0),
        N, M,
        BLOCK_SIZE=32
    )
    return out

# Reference implementation
def reference(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)

# Example usage
if __name__ == "__main__":
    # Create random input tensors
    x = torch.rand((1024, 512), device='cuda', dtype=torch.float16)
    w = torch.rand((1024,), device='cuda', dtype=torch.float16)
    
    # Compute using the reference implementation
    ref_output = reference(x, w)
    
    # Compute using the Triton implementation
    triton_output = triton_rmsnorm(x, w)
    
    # Check if the outputs are close
    assert torch.allclose(ref_output, triton_output, atol=1e-3, rtol=1e-3), "Outputs do not match"
    
    print("Reference and Triton outputs match!")
