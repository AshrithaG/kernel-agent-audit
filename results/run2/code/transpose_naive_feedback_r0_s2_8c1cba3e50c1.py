import torch
import triton
import triton.language as tl

# Define the Triton kernel for transposing a 2D tensor
@triton.jit
def transpose_kernel(
    x_ptr, y_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    **META
):
    # Define the program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the offsets for the current block
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Mask out-of-bound accesses
    mask = (rm < M) & (rn < N)

    # Load the data from the input tensor
    x = tl.load(x_ptr + rm[:, None] * stride_xm + rn[None, :] * stride_xn, mask=mask, other=0.0)

    # Store the data to the output tensor in transposed order
    tl.store(y_ptr + rn[:, None] * stride_ym + rm[None, :] * stride_yn, x, mask=mask)

# Define the kernel wrapper function
def kernel(x):
    # Get the shape of the input tensor
    M, N = x.shape

    # Define the block sizes
    BLOCK_M = 128
    BLOCK_N = 128

    # Create the output tensor
    y = torch.empty_like(x)

    # Launch the Triton kernel
    grid = lambda META: (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )
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
    x = torch.rand((1024, 2048), device='cuda', dtype=torch.float16)

    # Transpose the tensor using the Triton kernel
    y_triton = kernel(x)

    # Transpose the tensor using the reference implementation
    y_ref = x.t().contiguous()

    # Check if the results are the same
    assert torch.allclose(y_triton, y_ref, atol=1e-3, rtol=1e-3)

    print("Transpose successful!")
