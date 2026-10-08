import torch
import triton
import triton.language as tl
import math

@triton.jit
def rmsnorm_kernel(
    X, W, Out, 
    stride_xm, stride_xn, stride_ym, stride_yn,
    stride_w, stride_out,
    N, BLOCK_SIZE: tl.constexpr
):
    # Define the indices for the current thread
    m = tl.program_id(0)
    n = tl.load(X + m * stride_xm + tl.arange(0, BLOCK_SIZE) * stride_xn, mask=n < N, other=0.0)
    
    # Load the data into registers
    x = tl.load(X + m * stride_xm + n * stride_xn, mask=n < N, other=0.0)
    w = tl.load(W + n, mask=n < N, other=0.0)
    
    # Compute the sum of squares
    sum_squares = tl.sum(x * x, axis=1)
    
    # Compute the root mean square
    rms = tl.sqrt(sum_squares / N + 1e-6)
    
    # Normalize and scale by weight
    normalized_x = x / rms
    scaled_x = normalized_x * w
    
    # Store the result
    tl.store(Out + m * stride_ym + n * stride_yn, scaled_x, mask=n < N)

def kernel(X, W):
    # Ensure the inputs are float16
    X = X.half()
    W = W.half()
    
    # Get the shape of the input tensor
    M, N = X.shape
    
    # Define the grid and block size
    BLOCK_SIZE = 128
    grid = lambda META: (M, )
    
    # Allocate output tensor
    Out = torch.empty_like(X)
    
    # Launch the kernel
    rmsnorm_kernel[grid](X, W, Out,
                          X.stride(0), X.stride(1), Out.stride(0), Out.stride(1),
                          W.stride(0), Out.stride(0),
                          N, BLOCK_SIZE)
    
    return Out

# Example usage
if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    X = torch.randn(32, 512, device='cuda', dtype=torch.float16)
    W = torch.randn(512, device='cuda', dtype=torch.float16)
    
    # Reference implementation
    def reference(X, W):
        xf = X.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * W.float()).to(X.dtype)
    
    # Triton implementation
    out_ref = reference(X, W)
    out_triton = kernel(X, W)
    
    # Check if the outputs are the same
    assert torch.allclose(out_ref, out_triton, atol=1e-3, rtol=1e-3), "Outputs do not match"
    print("Outputs match within tolerance")
