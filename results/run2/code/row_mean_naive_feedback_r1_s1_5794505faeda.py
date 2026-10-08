import torch
import triton
import triton.language as tl

@triton.jit
def mean_kernel(x_ptr, out_ptr, N, M, BLOCK_SIZE: tl.constexpr):
    # Compute the row index for this thread
    row_idx = tl.program_id(axis=0)
    
    # Compute the start and end indices for the current row
    start = row_idx * M
    end = start + M
    
    # Load the elements of the current row into a vector
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    x = tl.load(x_ptr + start + offsets, mask=mask, other=0.0)
    
    # Compute the sum of the elements in the current row
    sum_x = tl.sum(x, axis=0)
    
    # Store the sum back to global memory
    tl.store(out_ptr + row_idx, sum_x)

def kernel(x):
    N, M = x.shape
    BLOCK_SIZE = 128
    out = torch.zeros(N, dtype=torch.float32, device=x.device)
    
    # Launch the kernel
    grid = lambda opt: (N,)
    mean_kernel[grid](x, out, N, M, BLOCK_SIZE)
    
    # Compute the mean by dividing the sum by M
    out /= M
    
    return out.to(x.dtype)

# Example usage:
# x = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
# result = kernel(x)
# print(result)
