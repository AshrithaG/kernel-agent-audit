import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel for ReLU activation
@triton.jit
def relu_kernel(x_ptr, out_ptr, stride_x, stride_out, N: tl.constexpr):
    # Get the program ID for the current threadblock
    pid = tl.program_id(axis=0)
    # Calculate the row index for the current threadblock
    row = pid * tl.block.size(0)
    # Create a mask to ensure we don't go out of bounds
    mask = row < N
    # Load the input values into registers
    x = tl.load(x_ptr + row * stride_x, mask=mask)
    # Apply ReLU activation
    x = tl.where(x > 0, x, 0)
    # Store the result back to global memory
    tl.store(out_ptr + row * stride_out, x, mask=mask)

# Define the Triton kernel for matrix multiplication and addition of bias
@triton.jit
def matmul_bias_kernel(a_ptr, b_ptr, bias_ptr, out_ptr, M, N, K, stride_a, stride_b, stride_bias, stride_out, BLOCK_SIZE: tl.constexpr):
    # Get the program ID for the current threadblock
    pid = tl.program_id(axis=0)
    # Calculate the row and column indices for the current threadblock
    row = pid // (N // BLOCK_SIZE)
    col = pid % (N // BLOCK_SIZE)
    # Create masks to ensure we don't go out of bounds
    mask_a = row * BLOCK_SIZE < M
    mask_b = col * BLOCK_SIZE < N
    mask_bias = col * BLOCK_SIZE < N
    mask_out = row * BLOCK_SIZE < M
    # Initialize the output value
    acc = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
    # Perform the matrix multiplication
    for k in range(K // BLOCK_SIZE):
        a = tl.load(a_ptr + row * BLOCK_SIZE * stride_a + k * BLOCK_SIZE, mask=mask_a[:, None])
        b = tl.load(b_ptr + k * BLOCK_SIZE * stride_b + col * BLOCK_SIZE, mask=mask_b[None, :])
        acc += tl.dot(a, b)
    # Add the bias
    bias = tl.load(bias_ptr + col * BLOCK_SIZE, mask=mask_bias)
    acc += bias
    # Apply ReLU activation
    acc = tl.where(acc > 0, acc, 0)
    # Store the result back to global memory
    tl.store(out_ptr + row * BLOCK_SIZE * stride_out + col * BLOCK_SIZE, acc, mask=mask_out[:, None])

# Define the Triton kernel wrapper function
def kernel(a, b, bias):
    # Get the shapes of the input tensors
    M, K = a.shape
    K, N = b.shape
    # Define the block size for the Triton kernel
    BLOCK_SIZE = 128
    # Launch the Triton kernel for matrix multiplication and addition of bias
    grid = lambda opt: (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
    matmul_bias_kernel[grid](a, b, bias, a, M, N, K, a.stride(0), b.stride(0), bias.stride(0), a.stride(0), BLOCK_SIZE)
    # Launch the Triton kernel for ReLU activation
    relu_kernel[a, N](a, a, a.stride(0), a.stride(0))
    # Return the result
    return a

# Example usage
if __name__ == "__main__":
    # Create example input tensors
    a = torch.randn(1024, 512, device="cuda", dtype=torch.float16)
    b = torch.randn(512, 256, device="cuda", dtype=torch.float16)
    bias = torch.randn(256, device="cuda", dtype=torch.float16)
    # Call the Triton kernel wrapper function
    result = kernel(a, b, bias)
    # Print the result
    print(result)
