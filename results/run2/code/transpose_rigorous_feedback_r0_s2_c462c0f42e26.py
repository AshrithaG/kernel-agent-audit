import torch
import triton
import triton.language as tl

@triton.jit
def transpose_kernel(
    x_ptr, y_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Compute row and column indices
    row = (pid // num_pid_n) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = (pid % num_pid_n) * BLOCK_N + tl.arange(0, BLOCK_N)

    # Bounds check
    mask = (row < M) & (col < N)

    # Load elements from input matrix
    x = tl.load(x_ptr + row[:, None] * N + col, mask=mask, other=0.0)

    # Transpose by swapping row and column indices
    y = tl.transpose(x)

    # Store elements to output matrix
    tl.store(y_ptr + col[:, None] * M + row, y, mask=mask)

# Function to transpose a 2-D tensor using Triton
def kernel(x):
    M, N = x.shape
    assert M <= 4096 and N <= 4096, "Matrix size exceeds supported limit"
    
    # Define block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    
    # Allocate output tensor
    y = torch.empty_like(x)
    
    # Launch Triton kernel
    grid = lambda META: (tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N),)
    transpose_kernel[grid](
        x.contiguous(), y.contiguous(),
        M, N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return y.contiguous()

# Example usage
if __name__ == "__main__":
    import time

    # Create a random 2-D tensor
    x = torch.randn(1024, 2048, device='cuda', dtype=torch.float16)

    # Measure performance of the reference implementation
    start_time = time.time()
    ref_y = x.t().contiguous()
    ref_time = time.time() - start_time
    print(f"Reference implementation time: {ref_time:.4f} seconds")

    # Measure performance of the Triton kernel
    start_time = time.time()
    triton_y = kernel(x)
    triton_time = time.time() - start_time
    print(f"Triton kernel time: {triton_time:.4f} seconds")

    # Verify correctness
    assert torch.allclose(ref_y, triton_y), "The outputs do not match"
