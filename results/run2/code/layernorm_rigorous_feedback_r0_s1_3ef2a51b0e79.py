import torch
import triton
import triton.language as tl
import math

# Triton kernel for LayerNorm
@triton.jit
def layernorm_kernel(
    X, W, B, Y,
    stride_xz, stride_yz,
    N, D,
    BLOCK_SIZE: tl.constexpr
):
    # Get the position of the current thread
    off_z = tl.program_id(0)
    off_d = tl.load(X + off_z * stride_xz + D - 1)
    
    # Load the data into shared memory
    x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(off_d, off_d + BLOCK_SIZE, BLOCK_SIZE):
        if i < D:
            x[i - off_d] = tl.load(X + off_z * stride_xz + i)
    
    # Compute mean and variance
    mean = tl.sum(x) / D
    var = tl.sum((x - mean) ** 2) / D
    
    # Normalize and scale
    epsilon = 1e-5
    inv_var = 1.0 / tl.sqrt(var + epsilon)
    x_hat = (x - mean) * inv_var
    y = x_hat * tl.load(W + off_d) + tl.load(B + off_d)
    
    # Store the result
    for i in range(off_d, off_d + BLOCK_SIZE, BLOCK_SIZE):
        if i < D:
            tl.store(Y + off_z * stride_yz + i, y[i - off_d])

# Triton wrapper function
def layernorm_triton(X, W, B):
    assert X.dtype == torch.float16 and W.dtype == torch.float16 and B.dtype == torch.float16
    N, D = X.shape
    Y = torch.empty_like(X)
    
    # Launch the Triton kernel
    grid = lambda META: (N,)
    layernorm_kernel[grid](
        X, W, B, Y,
        X.stride(0), Y.stride(0),
        N, D,
        BLOCK_SIZE=128
    )
    
    return Y

# Reference implementation for comparison
def layernorm_reference(X, W, B):
    return torch.nn.functional.layer_norm(X.float(), (X.shape[-1],), W.float(), B.float(), 1e-5).to(X.dtype)

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(128, 512, device='cuda', dtype=torch.float16)
    W = torch.randn(512, device='cuda', dtype=torch.float16)
    B = torch.randn(512, device='cuda', dtype=torch.float16)
    
    Y_ref = layernorm_reference(X, W, B)
    Y_triton = layernorm_triton(X, W, B)
    
    print("Reference Output:", Y_ref)
    print("Triton Output:", Y_triton)
    print("Max Difference:", torch.max(torch.abs(Y_ref - Y_triton)))
