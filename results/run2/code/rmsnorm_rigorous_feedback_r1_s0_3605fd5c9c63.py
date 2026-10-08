import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, N, M, BLOCK_SIZE: tl.constexpr):
    # Calculate the row and column indices
    row = tl.arange(0, BLOCK_SIZE)
    col = tl.program_id(axis=0)

    # Load data from global memory
    x = tl.load(x_ptr + row[:, None] + col[None, :], mask=row[:, None] < N, other=0.0)
    w = tl.load(w_ptr + col, mask=col < M, other=0.0)

    # Compute the square of each element
    x_squared = x * x

    # Sum the squares along the rows
    sum_squares = tl.sum(x_squared, axis=0)

    # Add epsilon to avoid division by zero
    sum_squares += 1e-6

    # Compute the inverse square root
    inv_sqrt = 1.0 / tl.sqrt(sum_squares)

    # Normalize the elements
    normalized_x = x * inv_sqrt

    # Scale by the weight
    scaled_x = normalized_x * w

    # Store the result back to global memory
    tl.store(out_ptr + row[:, None] + col[None, :], scaled_x, mask=row[:, None] < N)

# Define the Triton kernel wrapper
def kernel(x, w):
    N, M = x.shape
    BLOCK_SIZE = 128

    # Allocate output tensor
    out = torch.empty_like(x)

    # Launch the Triton kernel
    grid = lambda META: (
        M,
        triton.cdiv(N, BLOCK_SIZE),
    )
    rmsnorm_kernel[grid](x, w, out, N, M, BLOCK_SIZE=BLOCK_SIZE)

    return out

# Example usage
if __name__ == "__main__":
    # Create random input tensors
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    w = torch.randn(1024, device='cuda', dtype=torch.float16)

    # Run the Triton kernel
    result = kernel(x, w)

    # Verify the result against the reference implementation
    ref_result = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)

    # Check if the results are close
    assert torch.allclose(result, ref_result), "The Triton kernel result does not match the reference implementation."
    print("Kernel execution is correct.")
