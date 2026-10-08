import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel for adding bias and approximating GELU
@triton.jit
def add_bias_gelu_kernel(
    x_ptr, b_ptr, out_ptr, M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Define the program ID
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    # Calculate the row and column indices
    row = pid * BLOCK_M
    col = bid * BLOCK_N

    # Load the data into registers
    x = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)
    b = tl.zeros([BLOCK_N], dtype=tl.float16)

    # Load the bias vector
    offsets_b = col + tl.arange(0, BLOCK_N)
    mask_b = offsets_b < N
    tl.store(b, tl.load(b_ptr + offsets_b, mask=mask_b))

    # Load the matrix rows
    offsets_x = row * N + col + tl.arange(0, BLOCK_M)[:, None] * N + tl.arange(0, BLOCK_N)[None, :]
    mask_x = (offsets_x // N < M) & (offsets_x % N < N)
    x = tl.load(x_ptr + offsets_x, mask=mask_x)

    # Add the bias
    x += b[None, :]

    # Approximate GELU using tanh
    x = 0.5 * x * (1 + tl.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

    # Store the result back to global memory
    tl.store(out_ptr + offsets_x, x, mask=mask_x)

# Define the wrapper function to launch the Triton kernel
def kernel(x, b):
    M, N = x.shape
    assert x.dtype == torch.float16 and b.dtype == torch.float16, "Inputs must be float16"
    assert x.shape[1] == b.shape[0], "Bias vector must have the same length as the number of columns in x"

    # Allocate output tensor
    out = torch.empty_like(x)

    # Define grid and block sizes
    BLOCK_M = 128
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch the Triton kernel
    add_bias_gelu_kernel[grid](x, b, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    return out

# Example usage
if __name__ == "__main__":
    # Create example inputs
    x = torch.rand((1024, 1024), device='cuda', dtype=torch.float16)
    b = torch.rand((1024,), device='cuda', dtype=torch.float16)

    # Run the Triton kernel
    result = kernel(x, b)

    # Print the result
    print(result)
