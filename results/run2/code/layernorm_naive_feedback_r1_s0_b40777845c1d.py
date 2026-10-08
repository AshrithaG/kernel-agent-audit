import torch
import triton
import triton.language as tl

# Triton kernel for layer normalization
@triton.jit
def layer_norm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, mean_ptr, rsqrt_var_ptr, N, D, BLOCK_SIZE: tl.constexpr
):
    # Define the program ID and block start
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < D

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / N

    # Compute variance
    var = tl.sum((x - mean) ** 2, axis=0) / N
    rsqrt_var = 1.0 / tl.sqrt(var + 1e-5)

    # Normalize and scale
    y = (x - mean) * rsqrt_var
    y = y * tl.load(w_ptr + offsets, mask=mask, other=0.0) + tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Store results
    tl.store(y_ptr + offsets, y, mask=mask)
    tl.store(mean_ptr, mean)
    tl.store(rsqrt_var_ptr, rsqrt_var)

# Triton kernel wrapper
def kernel(x, w, b):
    N, D = x.shape
    BLOCK_SIZE = 128
    # Allocate output buffers
    y = torch.empty_like(x)
    mean = torch.empty((1,), device=x.device, dtype=torch.float32)
    rsqrt_var = torch.empty((1,), device=x.device, dtype=torch.float32)

    # Launch kernel
    grid = lambda meta: (triton.cdiv(D, BLOCK_SIZE),)
    layer_norm_kernel[grid](x, w, b, y, mean, rsqrt_var, N, D, BLOCK_SIZE=BLOCK_SIZE)

    return y

# Reference implementation for comparison
def reference(x, w, b):
    return torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float(), 1e-5).to(x.dtype)

# Example usage
if __name__ == "__main__":
    import torch

    # Random input data
    N, D = 1024, 512
    x = torch.randn(N, D, device='cuda', dtype=torch.float16)
    w = torch.randn(D, device='cuda', dtype=torch.float16)
    b = torch.randn(D, device='cuda', dtype=torch.float16)

    # Run Triton kernel
    y_triton = kernel(x, w, b)

    # Run reference implementation
    y_ref = reference(x, w, b)

    # Check correctness
    assert torch.allclose(y_triton, y_ref, atol=1e-3, rtol=1e-3)

    print("Triton kernel output matches reference implementation.")
