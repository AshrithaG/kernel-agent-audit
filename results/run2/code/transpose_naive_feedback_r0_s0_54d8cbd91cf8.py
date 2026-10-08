import torch
import triton
import triton.language as tl

@triton.jit
def transpose_kernel(input_ptr, output_ptr, M: tl.constexpr, N: tl.constexpr, stride_m_input: tl.constexpr, stride_n_input: tl.constexpr, stride_m_output: tl.constexpr, stride_n_output: tl.constexpr):
    # Define the block size
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 32
    
    # Calculate the position within the block
    row = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Ensure we don't go out of bounds
    mask = (row < M) & (col < N)
    
    # Load from input
    input_indices = row[:, None] * stride_m_input + col[None, :] * stride_n_input
    input_values = tl.load(input_ptr + input_indices, mask=mask, other=0.0)
    
    # Store to output
    output_indices = col[:, None] * stride_m_output + row[None, :] * stride_n_output
    tl.store(output_ptr + output_indices, input_values, mask=mask)

def kernel(x):
    M, N = x.shape
    # Allocate output tensor
    y = torch.empty_like(x).to(x.device)
    
    # Define grid size
    grid_size = (
        triton.cdiv(M, 128),
        triton.cdiv(N, 32),
    )
    
    # Launch the kernel
    transpose_kernel[grid_size](
        x.data_ptr(),
        y.data_ptr(),
        M,
        N,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
    )
    
    return y

# Example usage
if __name__ == "__main__":
    x = torch.randn((1024, 2048), device='cuda', dtype=torch.float16)
    y = kernel(x)
    print(y.shape)
    print(torch.allclose(y, x.t().contiguous()))
