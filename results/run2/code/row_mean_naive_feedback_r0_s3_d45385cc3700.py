import torch
import triton
import triton.language as tl
import math

@triton.jit
def mean_kernel(
    X,  # pointer to input matrix
    Y,  # pointer to output matrix
    N,  # number of rows
    M,  # number of columns
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread computes the sum of a row
    row_id = tl.program_id(0)
    if row_id >= N:
        return
    row_start = row_id * M
    row_end = row_start + M
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for col in range(row_start, row_end, BLOCK_SIZE):
        cols = tl.load(X + col, mask=(col < row_end), other=0.0)
        acc += cols
    # Sum the results of all threads in the block
    acc = tl.sum(acc, axis=0)
    # Store the result in the output matrix
    tl.store(Y + row_id, acc / M)

def kernel(X):
    N, M = X.shape
    # Allocate output tensor
    Y = torch.empty(N, dtype=torch.float32, device=X.device)
    # Define grid and block sizes
    BLOCK_SIZE = 1024
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
    # Launch the kernel
    mean_kernel[grid](X, Y, N, M, BLOCK_SIZE=BLOCK_SIZE)
    # Convert back to original dtype
    return Y.to(X.dtype)

# Example usage
if __name__ == "__main__":
    import time
    import numpy as np

    # Create a random 2D tensor of shape (N, M) with float16 values
    N, M = 1024, 512
    X = torch.tensor(np.random.rand(N, M), dtype=torch.float16, device='cuda')

    # Measure the performance of the reference implementation
    start_time = time.time()
    ref_result = X.float().mean(dim=-1).to(X.dtype)
    ref_time = time.time() - start_time
    print(f"Reference implementation time: {ref_time:.4f} seconds")

    # Measure the performance of the Triton kernel
    start_time = time.time()
    triton_result = kernel(X)
    triton_time = time.time() - start_time
    print(f"Triton kernel time: {triton_time:.4f} seconds")

    # Check if the results are the same within floating-point tolerance
    assert torch.allclose(ref_result, triton_result), "Results do not match"
    print("Results match within floating-point tolerance")
