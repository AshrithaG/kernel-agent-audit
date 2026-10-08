import torch
import triton
import triton.language as tl

# Triton kernel to compute mean over the last dimension
@triton.jit
def mean_kernel(x_ptr, out_ptr, n_elements_per_row, N, BLOCK_SIZE: tl.constexpr):
    # Get the row index
    row_idx = tl.program_id(0)
    # Calculate the start index for this row
    start_idx = row_idx * n_elements_per_row
    # Load elements into shared memory
    elements = tl.load(x_ptr + start_idx, mask=row_idx < N, other=0.0)
    # Sum the elements
    sum_elements = tl.sum(elements, axis=0)
    # Compute the mean
    mean = sum_elements / n_elements_per_row
    # Store the result
    tl.store(out_ptr + row_idx, mean, mask=row_idx < N)

def kernel(x):
    # Number of rows
    N = x.shape[0]
    # Number of elements per row
    n_elements_per_row = x.shape[1]
    # Output tensor
    out = torch.empty(N, dtype=x.dtype, device=x.device)
    # Block size
    BLOCK_SIZE = 128
    # Launch the kernel
    grid = lambda meta: (N,)
    mean_kernel[grid](x, out, n_elements_per_row, N, BLOCK_SIZE=BLOCK_SIZE)
    return out

# Example usage
if __name__ == "__main__":
    # Create a random float16 tensor
    x = torch.rand((1024, 512), dtype=torch.float16, device='cuda')
    # Compute row means using the Triton kernel
    result = kernel(x)
    # Verify the result
    reference_result = x.float().mean(dim=-1).to(x.dtype)
    assert torch.allclose(result, reference_result, atol=1e-3, rtol=1e-3)
    print("Kernel output matches reference implementation!")
