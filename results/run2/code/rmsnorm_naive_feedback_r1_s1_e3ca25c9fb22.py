import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_rows, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_id = pid // (BLOCK_SIZE // n_cols)
    col_id = (pid % (BLOCK_SIZE // n_cols)) * n_cols

    indices = col_id + tl.arange(0, BLOCK_SIZE // n_cols) * n_cols
    mask = indices < n_cols

    x = tl.load(x_ptr + row_id * n_cols + indices, mask=mask, other=0.0)
    w = tl.load(w_ptr + indices, mask=mask, other=0.0)

    squared_sum = tl.sum(x * x, axis=1)
    rms = tl.sqrt(squared_sum + eps)
    normalized_x = x / rms
    result = normalized_x * w

    tl.store(out_ptr + row_id * n_cols + indices, result, mask=mask)

def kernel(x, w, eps=1e-6):
    assert x.ndim == 2, "Input tensor must be 2-dimensional"
    assert w.ndim == 1, "Weight vector must be 1-dimensional"
    assert x.shape[1] == w.shape[0], "Last dimension of x must match length of w"

    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_rows * n_cols, BLOCK_SIZE),)
    rmsnorm_kernel[grid](x, w, out, n_rows, n_cols, eps, BLOCK_SIZE)

    return out

# Example usage:
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(128, 512, device='cuda', dtype=torch.float16)
    w = torch.randn(512, device='cuda', dtype=torch.float16)

    result = kernel(x, w)
    print(result)
