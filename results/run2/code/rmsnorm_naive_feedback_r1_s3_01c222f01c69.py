import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, N, eps, BLOCK_SIZE: tl.constexpr):
    # Each thread computes one element of the output
    idx = tl.program_id(axis=0)
    row_idx = idx // BLOCK_SIZE
    col_idx = idx % BLOCK_SIZE

    # Load the current row of x and the corresponding weight
    if row_idx < N:
        x = tl.load(x_ptr + row_idx * BLOCK_SIZE + col_idx, mask=col_idx < BLOCK_SIZE, other=0.0)
        w = tl.load(w_ptr + col_idx, mask=col_idx < BLOCK_SIZE, other=0.0)

        # Compute the sum of squares for the current row
        sum_squares = tl.sum(x * x, axis=0)

        # Compute the RMS and normalize
        rms = tl.sqrt(sum_squares / BLOCK_SIZE + eps)
        normalized_x = x / rms

        # Apply the weight
        result = normalized_x * w

        # Store the result
        tl.store(out_ptr + row_idx * BLOCK_SIZE + col_idx, result, mask=col_idx < BLOCK_SIZE)

def kernel(x, w, eps=1e-6):
    assert x.ndim == 2, "Input tensor must be 2-dimensional"
    assert w.ndim == 1, "Weight vector must be 1-dimensional"
    assert x.shape[1] == w.shape[0], "Last dimension of x must match length of w"

    N, D = x.shape
    BLOCK_SIZE = triton.next_power_of_2(D)

    out = torch.empty_like(x)
    grid = lambda meta: (N * BLOCK_SIZE,)
    rmsnorm_kernel[grid](x, w, out, N, eps, BLOCK_SIZE)

    return out

# Example usage:
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(128, 512, device='cuda', dtype=torch.float16)
    w = torch.randn(512, device='cuda', dtype=torch.float16)

    result = kernel(x, w)
    print(result)
