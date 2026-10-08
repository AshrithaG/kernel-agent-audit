import torch
import triton
import triton.language as tl

# Define the Triton kernel for transposing a 2D tensor
@triton.jit
def transpose_kernel(
    x_ptr, y_ptr, M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    stride_xm: tl.constexpr, stride_xn: tl.constexpr,
    stride_ym: tl.constexpr, stride_yn: tl.constexpr,
):
    # Get the program ID for the current thread
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Calculate the row and column indices for the current thread
    row = (pid // num_pid_n) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = (pid % num_pid_n) * BLOCK_N + tl.arange(0, BLOCK_N)

    # Ensure that we do not go out of bounds
    mask = (row < M) & (col < N)

    # Load elements from the input tensor
    x = tl.load(x_ptr + row[:, None] * stride_xm + col[None, :] * stride_xn, mask=mask, other=0.0)

    # Store elements into the output tensor (transposed)
    tl.store(y_ptr + col[:, None] * stride_ym + row[None, :] * stride_yn, x, mask=mask)

# Function to transpose a 2D tensor using the Triton kernel
def kernel(x):
    M, N = x.shape
    assert M <= 8192 and N <= 8192, "Matrix size exceeds the maximum supported by this kernel."

    # Define block sizes
    BLOCK_M = 128
    BLOCK_N = 128

    # Allocate output tensor
    y = torch.empty_like(x)

    # Launch the Triton kernel
    grid = lambda META: (tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N),)
    transpose_kernel[grid](
        x.data_ptr(), y.data_ptr(),
        M, N,
        BLOCK_M, BLOCK_N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
    )

    return y.contiguous()

# Example usage
if __name__ == "__main__":
    import time

    # Create a random 2D tensor
    x = torch.rand((4096, 4096), device="cuda", dtype=torch.float16)

    # Measure the time taken by the Triton kernel
    start_time = time.time()
    y = kernel(x)
    end_time = time.time()

    print(f"Time taken by Triton kernel: {end_time - start_time:.4f} seconds")

    # Verify the result
    reference_y = x.t().contiguous()
    assert torch.allclose(y, reference_y), "The output does not match the reference implementation."
    print("The output matches the reference implementation.")
