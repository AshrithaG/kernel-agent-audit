import torch
import triton
import triton.language as tl
import math

# Triton kernel for LayerNorm
@triton.jit
def layer_norm_kernel(
    X, W, B, Y, mean, rstd, N, D,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    # Compute the row and column indices for this thread
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = bid
    mask = row < N
    # Load data from X into registers
    x = tl.load(X + row * D + col, mask=mask, other=0.0)
    # Compute the mean
    mean_val = tl.sum(x, axis=0) / N
    tl.store(mean + col, mean_val)
    # Wait for all threads to compute the mean
    tl.barrier()
    # Load the mean value
    mean_val = tl.load(mean + col)
    # Compute the variance
    var = tl.sum((x - mean_val)**2, axis=0) / N
    # Compute the reciprocal of the standard deviation
    rstd_val = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(rstd + col, rstd_val)
    # Wait for all threads to compute the variance and rstd
    tl.barrier()
    # Load the rstd value
    rstd_val = tl.load(rstd + col)
    # Normalize the input
    y = (x - mean_val) * rstd_val
    # Scale and shift
    w = tl.load(W + col)
    b = tl.load(B + col)
    y = y * w + b
    # Store the result
    tl.store(Y + row * D + col, y, mask=mask)

# Triton kernel wrapper
def kernel(X, W, B):
    N, D = X.shape
    # Allocate output and temporary buffers
    Y = torch.empty_like(X)
    mean = torch.empty((D,), device=X.device, dtype=torch.float32)
    rstd = torch.empty((D,), device=X.device, dtype=torch.float32)
    # Define grid and block sizes
    BLOCK_SIZE = 128
    grid = lambda opt: (
        triton.cdiv(N, BLOCK_SIZE),
        D,
    )
    # Launch the kernel
    layer_norm_kernel[grid](X, W, B, Y, mean, rstd, N, D, BLOCK_SIZE=BLOCK_SIZE)
    return Y

# Reference implementation for validation
def reference(X, W, B):
    return torch.nn.functional.layer_norm(X.float(), (X.shape[-1],), W.float(), B.float(), 1e-5).to(X.dtype)

# Example usage
if __name__ == "__main__":
    # Create random input tensors
    N, D = 1024, 512
    X = torch.randn(N, D, device='cuda', dtype=torch.float16)
    W = torch.randn(D, device='cuda', dtype=torch.float16)
    B = torch.randn(D, device='cuda', dtype=torch.float16)
    
    # Run the kernel and reference implementation
    Y_triton = kernel(X, W, B)
    Y_ref = reference(X, W, B)
    
    # Check if the results are close
    assert torch.allclose(Y_triton, Y_ref, atol=1e-3, rtol=1e-3)
    print("Results match within tolerance!")
