import torch
import triton
import triton.language as tl

# Triton kernel to compute the mean over the last dimension of a 2-D tensor
@triton.jit
def mean_kernel(x_ptr, out_ptr, n, m, BLOCK_SIZE: tl.constexpr):
    # Define the program ID
    pid = tl.program_id(axis=0)
    
    # Compute the start and end indices for the current block
    start = pid * BLOCK_SIZE
    end = min(start + BLOCK_SIZE, n)
    
    # Initialize the sum for the current block
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load the elements from the input tensor into shared memory
    for i in range(start, end):
        x = tl.load(x_ptr + i * m, mask=i < n, other=0.0)
        sum_val[i - start] += x
    
    # Sum the values in the block
    sum_val = tl.sum(sum_val, axis=0)
    
    # Compute the mean
    mean_val = sum_val / m
    
    # Store the result in the output tensor
    if start < n:
        tl.store(out_ptr + start, mean_val)

# Triton wrapper function
def kernel(x):
    n, m = x.shape
    out = torch.empty(n, dtype=torch.float32, device=x.device)
    
    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    mean_kernel[grid](x, out, n, m, BLOCK_SIZE=1024)
    
    # Convert the output to the original dtype
    return out.to(x.dtype)

# Reference implementation for comparison
def reference(x):
    return x.float().mean(dim=-1).to(x.dtype)

# Example usage
if __name__ == "__main__":
    # Create a random float16 tensor
    x = torch.rand((1024, 512), dtype=torch.float16, device='cuda')
    
    # Compute the mean using the Triton kernel
    result_triton = kernel(x)
    
    # Compute the mean using the reference implementation
    result_ref = reference(x)
    
    # Check if the results are the same within floating-point tolerance
    assert torch.allclose(result_triton, result_ref), "The results do not match!"
    
    print("Results match!")
