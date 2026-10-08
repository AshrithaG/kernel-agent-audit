import torch
import triton
import triton.language as tl

@triton.jit
def swiglu_kernel(
    X, Y,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # Compute offsets for each thread
    off_m = tl.program_id(0) * BLOCK_M
    off_n = tl.program_id(1) * BLOCK_N
    
    # Load X into registers
    x = tl.load(X + off_m * stride_xm + off_n * stride_xn, mask=off_m < BLOCK_M, other=0.0)
    
    # Split x into two halves
    n = BLOCK_N // 2
    x1 = x[:n]
    x2 = x[n:]
    
    # Apply SiLU to the first half
    x1_silu = x1 * tl.sigmoid(x1)
    
    # Combine the results
    y = tl.cat([x1_silu, x2])
    
    # Store the result back to Y
    tl.store(Y + off_m * stride_ym + off_n * stride_yn, y, mask=off_m < BLOCK_M)

def kernel(X):
    # Get the shape and strides of the input tensor
    B, M, N = X.shape
    stride_xm = X.stride(1)
    stride_xn = X.stride(2)
    
    # Create an output tensor of the same shape
    Y = torch.empty_like(X)
    stride_ym = Y.stride(1)
    stride_yn = Y.stride(2)
    
    # Define grid and block sizes
    BLOCK_M = 32
    BLOCK_N = 128
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    
    # Launch the Triton kernel
    swiglu_kernel[grid_m, grid_n](
        X, Y,
        stride_xm, stride_xn,
        stride_ym, stride_yn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        device=X.device.index
    )
    
    return Y

# Example usage
if __name__ == "__main__":
    # Create a random input tensor of shape (B, M, N) with float16 dtype
    B, M, N = 1, 128, 256
    X = torch.rand((B, M, N), dtype=torch.float16, device='cuda')
    
    # Run the Triton kernel
    Y = kernel(X)
    
    # Verify the result against the reference implementation
    def reference(x):
        n = x.shape[-1] // 2
        return (torch.nn.functional.silu(x[..., :n].float()) * x[..., n:].float()).to(x.dtype)
    
    Y_ref = reference(X)
    
    # Check if the results are close
    assert torch.allclose(Y, Y_ref, atol=1e-3, rtol=1e-3), "The Triton kernel does not match the reference implementation."
    
    print("Triton kernel matches the reference implementation.")
