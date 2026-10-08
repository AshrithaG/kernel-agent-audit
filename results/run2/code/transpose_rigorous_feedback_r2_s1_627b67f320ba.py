import torch
import triton
import triton.language as tl

# Define the Triton kernel for transposing a 2D tensor
@triton.jit
def transpose_kernel(
    x_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    stride_xm, stride_xn, stride_ym, stride_yn
):
    # Define the program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Calculate the row and column indices
    row = pid // num_pid_n
    col = pid % num_pid_n
    
    # Load the input elements
    offsets_m = row * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    x = tl.load(x_ptr + offsets_m[:, None] * stride_xm + offsets_n[None, :] * stride_xn, mask=mask)
    
    # Store the output elements in transposed order
    offsets_ym = col * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_yn = row * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(y_ptr + offsets_ym[:, None] * stride_ym + offsets_yn[None, :] * stride_yn, x, mask=mask)

# Define the wrapper function to call the Triton kernel
def kernel(x):
    M, N = x.shape
    x_ptr = x.data_ptr()
    y = torch.empty_like(x)
    y_ptr = y.data_ptr()
    
    # Define the block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    
    # Launch the Triton kernel
    grid = lambda META: (tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N),)
    transpose_kernel[grid](
        x_ptr, y_ptr, M, N,
        BLOCK_M, BLOCK_N,
        x.stride(0), x.stride(1), y.stride(0), y.stride(1)
    )
    
    return y.contiguous()

# Example usage
if __name__ == "__main__":
    # Create a random float16 tensor
    x = torch.rand((1024, 1024), device='cuda', dtype=torch.float16)
    
    # Transpose the tensor using the Triton kernel
    y = kernel(x)
    
    # Verify the result
    reference_y = x.t().contiguous()
    assert torch.allclose(y, reference_y), "The transposed tensor does not match the reference"
    print("Transpose operation is correct.")
