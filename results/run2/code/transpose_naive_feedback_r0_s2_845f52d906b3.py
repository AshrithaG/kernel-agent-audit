import torch
import triton
import triton.language as tl

@triton.jit
def transpose_kernel(input_ptr, output_ptr, M: tl.constexpr, N: tl.constexpr, stride_xm: tl.constexpr, stride_xn: tl.constexpr, stride_ym: tl.constexpr, stride_yn: tl.constexpr):
    # Get the position of the current thread
    m = tl.program_id(0)
    n = tl.program_id(1)

    # Load the value from the input tensor
    if m < M and n < N:
        x = tl.load(input_ptr + m * stride_xm + n * stride_xn, mask=m < M & n < N)
        # Store the value in the transposed position in the output tensor
        tl.store(output_ptr + n * stride_ym + m * stride_yn, x, mask=m < M & n < N)

def kernel(x):
    M, N = x.shape
    output = torch.empty_like(x, memory_format=torch.contiguous_format)

    # Define grid and block dimensions
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch the kernel
    transpose_kernel[grid](x.data_ptr(), output.data_ptr(), M, N, x.stride(0), x.stride(1), output.stride(0), output.stride(1))

    return output

# Example usage:
if __name__ == "__main__":
    x = torch.randn((1024, 2048), device='cuda', dtype=torch.float16)
    transposed = kernel(x)
    print(transposed.shape)  # Should print torch.Size([2048, 1024])
    print(transposed.is_contiguous())  # Should print True
