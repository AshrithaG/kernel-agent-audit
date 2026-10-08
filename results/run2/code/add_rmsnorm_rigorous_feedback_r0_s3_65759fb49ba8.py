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
    BLOCK_SIZE = 32
    grid = (triton.cdiv(N, BLOCK_SIZE), H)
    rmsnorm_kernel[grid](
        X, R, W, X,
        N=N, H=H,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        device=X.device.index
    )
    return X

# Example usage
if __name__ == "__main__":
    X = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
    R = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
    W = torch.randn(512, dtype=torch.float16, device='cuda')

    # Reference implementation
    ref_output = (X.float() + R.float()) * torch.rsqrt((X.float() + R.float()).pow(2).mean(-1, keepdim=True) + 1e-6) * W.float()
    ref_output = ref_output.to(X.dtype)

    # Triton implementation
    output = X.clone()
    kernel(output, R, W)

    # Check if the outputs are close
    print(torch.allclose(ref_output, output))
