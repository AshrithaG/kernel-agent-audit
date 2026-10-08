import torch
import triton
import triton.language as tl

@triton.jit
def mean_kernel(
    X, Y, 
    stride_xm, stride_xn, 
    stride_ym, stride_yn,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the index of the row this program will operate on
    m = tl.program_id(0)
    
    # Load the entire row into shared memory
    x = tl.load(X + m * stride_xm, mask=m < N, other=0.0)
    
    # Compute the sum of the elements in the row
    sum_val = tl.sum(x)
    
    # Compute the mean by dividing the sum by the number of elements
    mean_val = sum_val / N
    
    # Store the result in the output tensor
    tl.store(Y + m * stride_ym, mean_val)

def kernel(X):
    M, N = X.shape
    assert X.is_cuda and X.dtype == torch.float16, "Input must be a float16 CUDA tensor"
    
    # Define the block size
    BLOCK_SIZE = 128
    
    # Allocate the output tensor
    Y = torch.empty((M,), device=X.device, dtype=torch.float16)
    
    # Launch the Triton kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE']),)
    mean_kernel[grid](X, Y, 
                      X.stride(0), X.stride(1), 
                      Y.stride(0), Y.stride(0),
                      N=N, BLOCK_SIZE=BLOCK_SIZE)
    
    return Y

# Example usage:
if __name__ == "__main__":
    import time
    import torch

    # Create a random float16 tensor
    X = torch.rand((1024, 512), device='cuda', dtype=torch.float16)

    # Measure the time taken by the reference implementation
    start_time = time.time()
    ref_result = X.float().mean(dim=-1).to(X.dtype)
    ref_time = time.time() - start_time
    print(f"Reference time: {ref_time:.6f} seconds")

    # Measure the time taken by the Triton kernel
    start_time = time.time()
    triton_result = kernel(X)
    triton_time = time.time() - start_time
    print(f"Triton time: {triton_time:.6f} seconds")

    # Check if the results are the same
    assert torch.allclose(ref_result, triton_result, atol=1e-3, rtol=1e-3), "Results do not match"
    print("Results match")
