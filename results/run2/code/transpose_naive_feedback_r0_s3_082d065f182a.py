import torch
import triton
import triton.language as tl

@triton.jit
def transpose_kernel(input_ptr, output_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Define the program ID
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute the row and column indices
    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Mask out-of-bounds accesses
    mask = (row < M)[:, None] & (col < N)[None, :]

    # Load data from the input tensor
    input_data = tl.load(input_ptr + row[:, None] * N + col[None, :], mask=mask, other=0.0)

    # Transpose the data
    transposed_data = tl.trans(input_data)

    # Store the transposed data to the output tensor
    tl.store(output_ptr + col[:, None] * M + row[None, :], transposed_data, mask=mask)

def kernel(input_tensor):
    M, N = input_tensor.shape
    output_tensor = torch.empty_like(input_tensor).transpose(0, 1).contiguous()

    # Define the grid
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch the kernel
    transpose_kernel[grid](input_tensor.data_ptr(), output_tensor.data_ptr(), M, N, BLOCK_M, BLOCK_N)

    return output_tensor

# Example usage
if __name__ == "__main__":
    import torch
    input_tensor = torch.randn((1024, 512), device='cuda', dtype=torch.float16)
    output_tensor = kernel(input_tensor)
    print(output_tensor.shape)
    print(output_tensor.dtype)
    print(torch.allclose(output_tensor, input_tensor.t().contiguous()))
