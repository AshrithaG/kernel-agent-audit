import torch
import triton
import triton.language as tl
import math

# Triton kernel for layer normalization
@triton.jit
def layer_norm_kernel(
    X, W, B, Y,
    N, C,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_w, stride_b,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # Define the block size and offsets
    off_m = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Load data from global memory to shared memory
    x = tl.load(X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn, mask=(off_m[:, None] < N) & (off_n[None, :] < C))
    
    # Compute mean
    mean = tl.sum(x, axis=1) / C
    
    # Compute variance
    var = tl.sum((x - mean[:, None]) ** 2, axis=1) / C
    
    # Compute normalized values
    x_hat = (x - mean[:, None]) / tl.sqrt(var[:, None] + 1e-5)
    
    # Apply weight and bias
    w = tl.load(W + off_n * stride_w, mask=off_n < C)
    b = tl.load(B + off_n * stride_b, mask=off_n < C)
    y = x_hat * w + b
    
    # Store results back to global memory
    tl.store(Y + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn, y, mask=(off_m[:, None] < N) & (off_n[None, :] < C))

# Wrapper function to call the Triton kernel
def kernel(X, W, B):
    # Get the shape of the input tensor
    N, C = X.shape
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 128
    
    # Launch the Triton kernel
    grid = (triton.cdiv(N, BLOCK_SIZE_M), triton.cdiv(C, BLOCK_SIZE_N))
    layer_norm_kernel[grid](
        X, W, B, X,
        N, C,
        X.stride(0), X.stride(1),
        X.stride(0), X.stride(1),
        W.stride(0), B.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return X

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
    W = torch.randn(512, device='cuda', dtype=torch.float16)
    B = torch.randn(512, device='cuda', dtype=torch.float16)
    
    Y_triton = kernel(X.clone(), W.clone(), B.clone())
    Y_ref = torch.nn.functional.layer_norm(X, (X.shape[-1],), W, B, 1e-5)
    
    print("Triton output:", Y_triton)
    print("Reference output:", Y_ref)
    print("Are outputs close?", torch.allclose(Y_triton, Y_ref))
