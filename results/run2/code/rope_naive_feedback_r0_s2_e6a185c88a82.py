import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel
@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, output_ptr,
    tokens, heads, dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # Get the position and head indices
    row_idx = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_idx = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Mask out-of-bound indices
    mask = (row_idx < tokens) & (col_idx < heads)
    
    # Calculate the half dimension
    d = dim // 2
    
    # Load x, cos, sin
    x1 = tl.load(x_ptr + row_idx[:, None] * heads * dim + col_idx[None, :] * dim, mask=mask[:, None] & (col_idx[None, :] < d), other=0.0)
    x2 = tl.load(x_ptr + row_idx[:, None] * heads * dim + col_idx[None, :] * dim + d, mask=mask[:, None] & (col_idx[None, :] < d), other=0.0)
    c = tl.load(cos_ptr + row_idx[:, None] * d, mask=mask[:, None], other=0.0)
    s = tl.load(sin_ptr + row_idx[:, None] * d, mask=mask[:, None], other=0.0)
    
    # Perform the rotation
    x1_rotated = x1 * c - x2 * s
    x2_rotated = x2 * c + x1 * s
    
    # Store the result
    tl.store(output_ptr + row_idx[:, None] * heads * dim + col_idx[None, :] * dim, x1_rotated, mask=mask[:, None])
    tl.store(output_ptr + row_idx[:, None] * heads * dim + col_idx[None, :] * dim + d, x2_rotated, mask=mask[:, None])

def kernel(x, cos, sin):
    # Ensure inputs are on the same device
    assert x.device == cos.device == sin.device, "All inputs must be on the same device"
    
    # Get the dimensions
    tokens, heads, dim = x.shape
    assert dim % 2 == 0, "Dimension must be even"
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Define grid and block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 32
    num_blocks_m = triton.cdiv(tokens, BLOCK_SIZE_M)
    num_blocks_n = triton.cdiv(heads, BLOCK_SIZE_N)
    
    # Launch the kernel
    rope_kernel[grid=(num_blocks_m, num_blocks_n), block=(BLOCK_SIZE_M, BLOCK_SIZE_N)](
        x.data_ptr(), cos.data_ptr(), sin.data_ptr(), output.data_ptr(),
        tokens, heads, dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output

# Example usage
if __name__ == "__main__":
    # Create some dummy data
    tokens = 16
    heads = 8
    dim = 64
    x = torch.rand((tokens, heads, dim), dtype=torch.float16, device='cuda')
    cos = torch.rand((tokens, dim // 2), dtype=torch.float16, device='cuda')
    sin = torch.rand((tokens, dim // 2), dtype=torch.float16, device='cuda')
    
    # Run the kernel
    result = kernel(x, cos, sin)
    print(result)
