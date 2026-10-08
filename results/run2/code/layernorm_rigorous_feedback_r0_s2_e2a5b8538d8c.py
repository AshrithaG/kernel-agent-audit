import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_kernel(x_ptr, w_ptr, b_ptr, out_ptr, N, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    # Calculate the row and column indices
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = bid

    # Load the input data
    x = tl.load(x_ptr + row[:, None] + col[None, :], mask=row[:, None] < N, other=0.0)

    # Compute the mean
    mean = tl.sum(x, axis=0) / N

    # Compute the variance
    var = tl.sum((x - mean)**2, axis=0) / N

    # Compute the normalized values
    inv_var = tl.rsqrt(var + eps)
    norm_x = (x - mean) * inv_var

    # Apply weight and bias
    w = tl.load(w_ptr + col, mask=col < N, other=0.0)
    b = tl.load(b_ptr + col, mask=col < N, other=0.0)
    out = norm_x * w + b

    # Store the output
    tl.store(out_ptr + row[:, None] + col[None, :], out, mask=row[:, None] < N)

# Triton kernel wrapper function
def kernel(x, w, b):
    N = x.shape[-1]
    assert x.shape[-1] == w.shape[0] == b.shape[0], "Last dimension of x must match dimensions of w and b"
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Define grid and block sizes
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE), x.shape[0])
    
    # Launch the kernel
    layernorm_kernel[grid](x, w, b, out, N, 1e-5, BLOCK_SIZE)
    
    return out

# Example usage
if __name__ == "__main__":
    import torch.nn.functional as F
    
    # Create random input tensors
    x = torch.randn(32, 1024, device='cuda', dtype=torch.float16)
    w = torch.randn(1024, device='cuda', dtype=torch.float16)
    b = torch.randn(1024, device='cuda', dtype=torch.float16)
    
    # Reference implementation
    ref_out = F.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float(), 1e-5).to(x.dtype)
    
    # Triton implementation
    triton_out = kernel(x, w, b)
    
    # Check if outputs are close
    print(torch.allclose(ref_out, triton_out))
