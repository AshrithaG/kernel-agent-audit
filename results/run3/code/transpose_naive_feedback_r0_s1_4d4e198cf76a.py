import torch
import triton
import triton.language as tl

@triton.jit
def _transpose_kernel(input_ptr, output_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Calculate the row and column indices for this block
    row = tl.program_id(0) * BLOCK_M
    col = tl.program_id(1) * BLOCK_N
    
    # Create a mask to ensure we don't read/write out of bounds
    mask = (row + tl.arange(0, BLOCK_M) < M) & (col + tl.arange(0, BLOCK_N) < N)
    
    # Load the input data into registers
    input_data = tl.load(input_ptr + (row + tl.arange(0, BLOCK_M))[:, None] * N + (col + tl.arange(0, BLOCK_N)), mask=mask, other=0.0).to(tl.float32)
    
    # Transpose the data in registers
    output_data = input_data.T
    
    # Store the transposed data back to the output pointer
    tl.store(output_ptr + (col + tl.arange(0, BLOCK_N))[:, None] * M + (row + tl.arange(0, BLOCK_M)), output_data, mask=mask)

def kernel(x):
    # Ensure the input is contiguous
    x = x.contiguous()
    
    # Get the dimensions of the input tensor
    M, N = x.shape
    
    # Calculate the number of blocks needed
    BLOCK_M = triton.next_power_of_2(M)
    BLOCK_N = triton.next_power_of_2(N)
    num_blocks_m = (M + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (N + BLOCK_N - 1) // BLOCK_N
    
    # Allocate the output tensor
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    
    # Launch the kernel
    grid = (num_blocks_m, num_blocks_n)
    _transpose_kernel[grid](x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    
    return out
