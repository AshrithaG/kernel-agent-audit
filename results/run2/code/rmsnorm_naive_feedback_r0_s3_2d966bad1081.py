import torch
import triton
import triton.language as tl
import math

# Triton kernel for RMSNorm
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n, eps, BLOCK_SIZE: tl.constexpr):
    # Define the thread index
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.thread_idx(axis=0)
    # Load the data into registers
    x = tl.load(x_ptr + idx, mask=(idx < n), other=0.0)
    # Compute the square of x
    x_sq = x * x
    # Sum the squares of x in each block
    sum_x_sq = tl.sum(x_sq, axis=0)
    # Compute the mean of the squares
    mean_x_sq = sum_x_sq / n
    # Add epsilon to avoid division by zero
    mean_x_sq += eps
    # Compute the reciprocal square root of the mean of squares
    rsqrt_mean_x_sq = 1.0 / tl.sqrt(mean_x_sq)
    # Load the weight
    w = tl.load(w_ptr + idx, mask=(idx < n), other=0.0)
    # Normalize x and multiply by the weight
    out = x * rsqrt_mean_x_sq * w
    # Store the result
    tl.store(out_ptr + idx, out, mask=(idx < n))

# Wrapper function to call the Triton kernel
def kernel(x, w):
    # Ensure the input tensors are on the same device
    assert x.device == w.device, "Input tensors must be on the same device"
    # Ensure the input tensors are of the same dtype
    assert x.dtype == w.dtype, "Input tensors must have the same dtype"
    # Ensure the input tensors are 2D
    assert x.ndim == 2, "Input tensor x must be 2D"
    assert w.ndim == 2, "Input tensor w must be 2D"
    # Ensure the last dimension of x and w are the same
    assert x.shape[1] == w.shape[1], "The last dimension of x and w must be the same"
    
    # Get the number of elements in the last dimension
    n = x.shape[1]
    # Get the number of elements in the batch dimension
    batch_size = x.shape[0]
    
    # Create an output tensor with the same shape and dtype as x
    out = torch.empty_like(x)
    
    # Define the block size
    BLOCK_SIZE = 128
    
    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(n, BLOCK_SIZE), batch_size)
    rmsnorm_kernel[grid](x, w, out, n, 1e-6, BLOCK_SIZE)
    
    return out

# Reference implementation for testing
def reference(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)

# Test the kernel
if __name__ == "__main__":
    # Create random input tensors
    x = torch.rand((1024, 512), device="cuda", dtype=torch.float16)
    w = torch.rand((1024, 512), device="cuda", dtype=torch.float16)
    
    # Run the kernel and reference implementation
    y = kernel(x, w)
    y_ref = reference(x, w)
    
    # Check if the outputs are the same
    assert torch.allclose(y, y_ref, atol=1e-3, rtol=1e-3), "The outputs are not the same"
    
    print("Test passed!")
