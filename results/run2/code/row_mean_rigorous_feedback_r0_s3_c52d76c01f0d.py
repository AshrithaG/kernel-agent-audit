import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel
@triton.jit
def mean_kernel(x_ptr, out_ptr, N: tl.constexpr, M: tl.constexpr):
    # Each thread computes the sum of a row
    row_id = tl.program_id(0)
    if row_id >= N:
        return
    row_sum = tl.zeros([M], dtype=tl.float32)
    for col_id in range(M):
        row_sum += tl.load(x_ptr + row_id * M + col_id, mask=col_id < M)
    # Compute the mean by dividing the sum by the number of elements
    row_mean = row_sum / M
    # Store the result
    tl.store(out_ptr + row_id, row_mean)

# Define the kernel wrapper function
def kernel(x):
    N, M = x.shape
    # Allocate output tensor
    out = torch.empty(N, dtype=torch.float16, device=x.device)
    # Launch the kernel
    grid = lambda META: (N,)
    mean_kernel[grid](x, out, N=N, M=M)
    return out

# Example usage
if __name__ == "__main__":
    # Create a random float16 tensor
    x = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
    # Compute the row mean using the Triton kernel
    result = kernel(x)
    print(result)
