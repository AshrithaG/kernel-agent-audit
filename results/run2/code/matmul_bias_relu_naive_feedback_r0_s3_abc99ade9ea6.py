import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel for ReLU activation
@triton.jit
def relu_kernel(x_ptr, out_ptr, stride_x, stride_out, n_elements):
    pid = tl.program_id(axis=0)
    offsets = range(pid * 32, min((pid + 1) * 32, n_elements))
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements, other=0.0)
    out = tl.where(x > 0, x, 0.0)
    tl.store(out_ptr + offsets, out, mask=offsets < n_elements)

# Triton kernel for matrix multiplication and addition of bias
@triton.jit
def matmul_bias_kernel(a_ptr, b_ptr, bias_ptr, out_ptr, M, N, K, stride_a, stride_b, stride_bias, stride_out):
    # Get the thread ID
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # Load the bias into shared memory
    bias = tl.load(bias_ptr + col)
    
    # Initialize the output value
    acc = tl.zeros([K], dtype=tl.float16)
    
    # Iterate over the K dimension
    for k in range(K):
        a = tl.load(a_ptr + row * stride_a + k)
        b = tl.load(b_ptr + col * stride_b + k)
        acc += a * b
    
    # Add the bias and apply ReLU
    acc += bias
    acc = tl.where(acc > 0, acc, 0.0)
    
    # Store the result
    tl.store(out_ptr + row * stride_out + col, acc)

# Triton kernel wrapper function
def kernel(a, b, bias):
    M, K = a.shape
    K, N = b.shape
    assert K == bias.shape[0], "Bias shape must match the second dimension of b"
    
    # Allocate output tensor
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Define grid and block dimensions
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))
    block = (16, 16)
    
    # Launch the Triton kernel
    matmul_bias_kernel[grid](a, b, bias, out, M, N, K, a.stride(0), b.stride(0), bias.stride(0), out.stride(0))
    
    return out

# Reference implementation for verification
def reference(a, b, bias):
    return torch.relu(a.float() @ b.float() + bias.float()).to(a.dtype)

# Example usage
if __name__ == "__main__":
    # Create random tensors
    M, K, N = 256, 512, 128
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    bias = torch.randn(N, device='cuda', dtype=torch.float16)
    
    # Run the Triton kernel and reference implementation
    out_triton = kernel(a, b, bias)
    out_reference = reference(a, b, bias)
    
    # Check if the results are the same
    assert torch.allclose(out_triton, out_reference, atol=1e-3, rtol=1e-3), "Results do not match"
    
    print("Results match!")
