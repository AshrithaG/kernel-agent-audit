import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    X, R, W, Out,
    N: tl.constexpr, H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    # Calculate the row and column indices
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = bid
    mask = row < N
    # Load the data
    x = tl.load(X + row[:, None] * H + col, mask=mask[:, None])
    r = tl.load(R + row[:, None] * H + col, mask=mask[:, None])
    # Add residual
    h = x + r
    # Compute the mean of squares
    sum_squares = tl.sum(h * h, axis=0)
    mean_squares = sum_squares / N
    # Compute the denominator
    denom = tl.sqrt(mean_squares + 1e-6)
    # Normalize and scale
    w = tl.load(W + col)
    out = h / denom * w
    # Store the result
    tl.store(Out + row[:, None] * H + col, out, mask=mask[:, None])

def kernel(X, R, W):
    N, H = X.shape
    # Ensure the inputs are float16
    X = X.to(torch.float16)
    R = R.to(torch.float16)
    W = W.to(torch.float16)
    # Allocate output tensor
    Out = torch.empty_like(X)
    # Define the grid and block sizes
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE), H)
    # Launch the kernel
    rmsnorm_kernel[grid](X, R, W, Out, N=N, H=H, BLOCK_SIZE=BLOCK_SIZE)
    return Out

# Example usage
if __name__ == "__main__":
    import math
    torch.manual_seed(0)
    N, H = 1024, 512
    X = torch.randn(N, H, device='cuda', dtype=torch.float16)
    R = torch.randn(N, H, device='cuda', dtype=torch.float16)
    W = torch.randn(H, device='cuda', dtype=torch.float16)

    # Reference implementation
    ref_out = (X.float() + R.float()) * torch.rsqrt((X.float() + R.float()).pow(2).mean(-1, keepdim=True) + 1e-6) * W.float().view(1, -1)
    ref_out = ref_out.to(X.dtype)

    # Triton implementation
    triton_out = kernel(X, R, W)

    # Check correctness
    assert torch.allclose(ref_out, triton_out, atol=1e-3, rtol=1e-3)
    print("Kernel output matches reference implementation.")
