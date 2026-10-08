import torch
import triton
import triton.language as tl

# Define the Triton kernel for transposing a 2D tensor
@triton.jit
def transpose_kernel(input_ptr, output_ptr, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Define the program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate the row and column indices
    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Mask out-of-bounds accesses
    mask = (row < M) & (col < N)

    # Load the input values
    input_values = tl.load(input_ptr + row[:, None] * stride_in_m + col[None, :] * stride_in_n, mask=mask, other=0.0)

    # Store the transposed values
    tl.store(output_ptr + col[:, None] * stride_out_m + row[None, :] * stride_out_n, input_values, mask=mask)

# Define the kernel wrapper function
def kernel(input):
    M, N = input.shape
    output = torch.empty_like(input)

    # Define the grid
    BLOCK_M = 128
    BLOCK_N = 128
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)

    # Launch the kernel
    transpose_kernel[grid_m, grid_n](
        input.contiguous().data_ptr(),
        output.data_ptr(),
        M,
        N,
        input.stride(0),
        input.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return output

# Example usage
if __name__ == "__main__":
    input_tensor = torch.rand((1024, 512), device='cuda', dtype=torch.float16)
    transposed_tensor = kernel(input_tensor)
    print(transposed_tensor.shape)
    print(transposed_tensor.dtype)
