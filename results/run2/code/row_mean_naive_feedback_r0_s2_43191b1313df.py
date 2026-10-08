import torch
import triton
import triton.language as tl

@triton.jit
def mean_kernel(
    x_ptr, output_ptr, 
    stride_xm, stride_xn, 
    N, M, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    cols_per_pid = tl.cdiv(N, BLOCK_SIZE)
    col_range = pid * cols_per_pid + tl.arange(0, cols_per_pid)
    mask = col_range < N
    col_idx = col_range[mask]
    x = tl.load(x_ptr + col_idx[:, None] * stride_xn, mask=mask[:, None], other=0.0)
    mean = tl.sum(x, axis=1) / M
    tl.store(output_ptr + col_idx, mean, mask=mask)

def kernel(x):
    M, N = x.shape
    output = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK_SIZE = 128
    grid = lambda META: (tl.cdiv(N, BLOCK_SIZE),)
    mean_kernel[grid](x, output, x.stride(0), x.stride(1), N, M, BLOCK_SIZE=BLOCK_SIZE)
    return output.to(x.dtype)

# Example usage:
if __name__ == "__main__":
    import torch
    x = torch.rand((1024, 512), device='cuda', dtype=torch.float16)
    result = kernel(x)
    print(result)
