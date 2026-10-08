import torch
import triton
import triton.language as tl
import math

# Triton kernel for row-wise softmax
@triton.jit
def softmax_kernel(
    X,  # input matrix
    Y,  # output matrix
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    m = tl.program_id(0)
    # Get the column index
    n = tl.load(X + m * stride_xm + tl.arange(0, BLOCK_SIZE) * stride_xn)
    # Load the row into shared memory
    x = tl.load(X + m * stride_xm + tl.arange(0, BLOCK_SIZE) * stride_xn, mask=n < BLOCK_SIZE)
    # Compute the max value in the row
    max_x = tl.max(x, axis=0)
    # Subtract the max value from each element to avoid overflow
    x = x - max_x
    # Compute the exponential of each element
    exp_x = tl.exp(x)
    # Compute the sum of the exponentials
    sum_exp_x = tl.sum(exp_x, axis=0)
    # Compute the softmax
    softmax_x = exp_x / sum_exp_x
    # Store the result back to global memory
    tl.store(Y + m * stride_ym + tl.arange(0, BLOCK_SIZE) * stride_yn, softmax_x, mask=n < BLOCK_SIZE)

# Triton wrapper function
def kernel(X):
    # Get the shape of the input tensor
    M, N = X.shape
    # Define the block size
    BLOCK_SIZE = 128
    # Create the output tensor
    Y = torch.empty_like(X)
    # Launch the Triton kernel
    grid = lambda META: (M,)
    softmax_kernel[grid](
        X, Y,
        X.stride(0), X.stride(1),
        Y.stride(0), Y.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return Y

# Example usage
if __name__ == "__main__":
    import time
    import torch.nn.functional as F

    # Set the device to CUDA
    device = torch.device("cuda")

    # Create a random input tensor
    X = torch.randn(1024, 512, dtype=torch.float16, device=device)

    # Reference implementation
    start_time = time.time()
    ref_Y = F.softmax(X, dim=-1)
    ref_time = time.time() - start_time
    print(f"Reference time: {ref_time:.6f} seconds")

    # Triton implementation
    start_time = time.time()
    triton_Y = kernel(X)
    triton_time = time.time() - start_time
    print(f"Triton time: {triton_time:.6f} seconds")

    # Check the results
    assert torch.allclose(ref_Y, triton_Y, atol=1e-3, rtol=1e-3)
    print("Results match within tolerance")
