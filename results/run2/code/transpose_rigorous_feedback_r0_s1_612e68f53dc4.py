import torch
import triton
import triton.language as tl

# Define the Triton kernel for transposing a 2D tensor
@triton.jit
def transpose_kernel(
    x_ptr, y_ptr,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    stride_xm: tl.constexpr, stride_xn: tl.constexpr,
    stride_ym: tl.constexpr, stride_yn: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Compute the row and column indices
    row = pid // num_pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
    col = pid % num_pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create a mask to ensure we do not go out of bounds
    mask = (row < M) & (col < N)

    # Load elements from the input tensor
    x = tl.load(x_ptr + row[:, None] * stride_xm + col[None, :] * stride_xn, mask=mask, other=0.0)

    # Store elements to the output tensor in transposed order
    tl.store(y_ptr + col[:, None] * stride_ym + row[None, :] * stride_yn, x, mask=mask)

# Define the wrapper function to handle the kernel launch
def kernel(x):
    # Get the shape of the input tensor
    M, N = x.shape

    # Define the block size
    BLOCK_M = 32
    BLOCK_N = 32

    # Allocate memory for the output tensor
    y = torch.empty_like(x)

    # Compute the grid size
    grid = lambda META: (tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N),)

    # Launch the kernel
    transpose_kernel[grid](
        x.contiguous(), y.contiguous(),
        M, N,
        BLOCK_M, BLOCK_N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
    )

    return y

# Example usage
if __name__ == "__main__":
    # Create a random float16 tensor
    x = torch.randn((1024, 2048), device='cuda', dtype=torch.float16)

    # Transpose the tensor using the Triton kernel
    y = kernel(x)

    # Verify the result
    reference_y = x.t().contiguous()
    assert torch.allclose(y, reference_y, atol=1e-3, rtol=1e-3), "The transposed tensor does not match the reference implementation."

    print("Transpose operation completed successfully.")
