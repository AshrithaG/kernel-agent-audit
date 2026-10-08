import torch
import triton
import triton.language as tl

@triton.jit
def _transpose_kernel(input_ptr, output_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Calculate the row and column indices for this block
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # Calculate the offsets for loading and storing
    row_offset = row * BLOCK_M
    col_offset = col * BLOCK_N
    
    # Load the data from the input matrix
    # We need to transpose the data, so we load columns and store rows
    x = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, BLOCK_M):
        for j in range(0, BLOCK_N):
            if row_offset + i < M and col_offset + j < N:
                x[i, j] = tl.load(input_ptr + (col_offset + j) * M + (row_offset + i)).to(tl.float32)
    
    # Store the transposed data into the output matrix
    for i in range(0, BLOCK_M):
        for j in range(0, BLOCK_N):
            if row_offset + i < M and col_offset + j < N:
                tl.store(output_ptr + (row_offset + i) * N + (col_offset + j), x[i, j].to(output_ptr.dtype.element_ty))

def kernel(x):
    x = x.contiguous()
    M, N = x.shape
    # Determine the block size
    BLOCK_M = triton.next_power_of_2(M)
    BLOCK_N = triton.next_power_of_2(N)
    
    # Allocate the output tensor
    out = torch.empty_like(x)
    
    # Launch the kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _transpose_kernel[grid](x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    
    return out
