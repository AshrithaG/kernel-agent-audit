import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel
@triton.jit
def bias_gelu_kernel(
    x_ptr, b_ptr, out_ptr, M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Define the position of the element to work on
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    block_pid_m = pid // num_pid_n
    block_pid_n = pid % num_pid_n

    # Calculate the row and column indices
    row = block_pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col = block_pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load the data from the input tensors
    x = tl.load(x_ptr + row[:, None] * N + col, mask=row[:, None] < M & col < N, other=0.0)
    b = tl.load(b_ptr + col, mask=col < N, other=0.0)

    # Add the bias to the input
    x = x + b

    # Apply the tanh-approximated GELU function
    cdf = 0.5 * (1.0 + tl.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * tl.pow(x, 3))))
    out = x * cdf

    # Store the result back to the output tensor
    tl.store(out_ptr + row[:, None] * N + col, out, mask=row[:, None] < M & col < N)

# Define the wrapper function
def kernel(x, b):
    M, N = x.shape
    # Ensure the input tensors are on the same device
    assert x.device == b.device, "Input tensors must be on the same device"
    # Ensure the input tensors are of type float16
    assert x.dtype == torch.float16, "Input tensor x must be of type float16"
    assert b.dtype == torch.float16, "Input tensor b must be of type float16"
    # Create the output tensor
    out = torch.empty_like(x)
    # Define the grid size
    BLOCK_M = 128
    BLOCK_N = 128
    grid = lambda META: (tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N),)
    # Launch the kernel
    bias_gelu_kernel[grid](x, b, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out

# Example usage
if __name__ == "__main__":
    # Create example input tensors
    x = torch.randn((1024, 512), dtype=torch.float16, device="cuda")
    b = torch.randn(512, dtype=torch.float16, device="cuda")
    # Call the kernel
    out = kernel(x, b)
    # Print the output tensor
    print(out)
