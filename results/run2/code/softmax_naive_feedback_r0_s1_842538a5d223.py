import torch
import triton
import triton.language as tl
import math

# Triton kernel for row-wise softmax
@triton.jit
def softmax_kernel(
    X,  # input pointer to the first element of the input tensor
    Y,  # output pointer to the first element of the output tensor
    stride_xm,  # stride of the input tensor along the m dimension
    stride_xn,  # stride of the input tensor along the n dimension
    stride_ym,  # stride of the output tensor along the m dimension
    stride_yn,  # stride of the output tensor along the n dimension
    N,  # number of elements in the n dimension
    BLOCK_SIZE: tl.constexpr,  # number of threads per block
):
    # Get the thread index
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    
    # Calculate the row index
    row_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load the row from the input tensor
    row = tl.load(X + row_idx[:, None] * stride_xn + bid * stride_xm, mask=row_idx < N, other=-float('inf'))
    
    # Compute the maximum value in the row
    row_max = tl.max(row, axis=1)
    
    # Subtract the maximum value from each element in the row
    row = row - row_max[:, None]
    
    # Compute the exponentials
    exp_row = tl.exp(row)
    
    # Compute the sum of the exponentials
    sum_exp_row = tl.sum(exp_row, axis=1)
    
    # Divide the exponentials by the sum to get the softmax values
    softmax_row = exp_row / sum_exp_row[:, None]
    
    # Store the result in the output tensor
    tl.store(Y + row_idx[:, None] * stride_yn + bid * stride_ym, softmax_row, mask=row_idx < N)

# Triton wrapper function
def softmax_triton(X):
    # Ensure the input is a float16 tensor
    if X.dtype != torch.float16:
        raise ValueError("Input tensor must be of type float16")
    
    # Get the shape of the input tensor
    m, n = X.shape
    
    # Allocate memory for the output tensor
    Y = torch.empty_like(X)
    
    # Define the block size
    BLOCK_SIZE = 128
    
    # Launch the Triton kernel
    grid = (math.ceil(m / BLOCK_SIZE), n)
    softmax_kernel[grid](X, Y, X.stride(0), X.stride(1), Y.stride(0), Y.stride(1), n, BLOCK_SIZE)
    
    return Y

# Reference implementation for comparison
def softmax_reference(X):
    return torch.softmax(X.float(), dim=-1).to(X.dtype)

# Example usage
if __name__ == "__main__":
    # Create a random float16 tensor
    X = torch.rand((1024, 512), dtype=torch.float16, device='cuda')
    
    # Compute softmax using the reference implementation
    Y_ref = softmax_reference(X)
    
    # Compute softmax using the Triton implementation
    Y_triton = softmax_triton(X)
    
    # Check if the results are the same within floating-point tolerance
    assert torch.allclose(Y_ref, Y_triton, atol=1e-3, rtol=1e-3), "The results do not match"
    
    print("Softmax computation is correct.")
