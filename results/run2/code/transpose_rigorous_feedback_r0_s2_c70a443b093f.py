import torch
import triton
import triton.language as tl

# Define the Triton kernel for transposing a 2D tensor
@triton.jit
def transpose_kernel(
    A, B,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Calculate the position of the element within the block
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    # Mask out elements that are out of bounds
    mask = (m < M)[:, None] & (n < N)[None, :]

    # Load elements from A
    a = tl.load(A + m[:, None] * stride_am + n[None, :] * stride_an, mask=mask)

    # Store elements into B
    tl.store(B + n[:, None] * stride_bm + m[None, :] * stride_bn, a, mask=mask)

# Function to transpose a 2D tensor using the Triton kernel
def kernel(A):
    M, N = A.shape
    A_data = A.contiguous().half().data_ptr()
    B_data = torch.empty_like(A).contiguous().half().data_ptr()

    # Define grid and block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)

    # Launch the Triton kernel
    transpose_kernel[grid_m, grid_n](
        A_data, B_data,
        M, N,
        A.stride(0), A.stride(1),
        B_data.stride(0), B_data.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )

    return torch.tensor_from_file(B_data, shape=(N, M), dtype=torch.float16).contiguous()

# Example usage
if __name__ == "__main__":
    A = torch.rand((1024, 2048), device="cuda", dtype=torch.float16)
    B = kernel(A)
    C = A.t().contiguous()
    print(torch.allclose(B, C))
