import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(x_ptr, r_ptr, w_ptr, out_ptr, stride_x, stride_r, stride_w, stride_out, N, H, BLOCK_SIZE: tl.constexpr):
    # Define the block size
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load inputs
    x = tl.load(x_ptr + offsets[:, None] * stride_x + tl.arange(0, H)[None, :], mask=mask[:, None], other=0.0)
    r = tl.load(r_ptr + offsets[:, None] * stride_r + tl.arange(0, H)[None, :], mask=mask[:, None], other=0.0)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)

    # Compute the residual
    h = x + r

    # Compute the mean of squares
    sum_squares = tl.sum(h**2, axis=1)
    mean_squares = sum_squares / H

    # Compute the RMSNorm
    rms = tl.sqrt(mean_squares + 1e-6)
    inv_rms = 1.0 / rms
    normalized_h = h * inv_rms[:, None]

    # Scale by w
    scaled_h = normalized_h * w[:, None]

    # Store the result
    tl.store(out_ptr + offsets[:, None] * stride_out + tl.arange(0, H)[None, :], scaled_h, mask=mask[:, None])

# Function to launch the kernel
def kernel(x, r, w):
    assert x.is_cuda and r.is_cuda and w.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == torch.float16 and r.dtype == torch.float16 and w.dtype == torch.float16, "Inputs must be float16"
    assert x.shape == r.shape and x.shape[:-1] == w.shape, "Incompatible shapes"

    N, H = x.shape
    BLOCK_SIZE = 128

    # Allocate output tensor
    out = torch.empty_like(x)

    # Launch the kernel
    grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)
    rmsnorm_kernel[grid](x, r, w, out, x.stride(0), x.stride(1), w.stride(0), out.stride(0), N, H, BLOCK_SIZE=BLOCK_SIZE)

    return out

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn((128, 512), device='cuda', dtype=torch.float16)
    r = torch.randn((128, 512), device='cuda', dtype=torch.float16)
    w = torch.randn((128,), device='cuda', dtype=torch.float16)

    out = kernel(x, r, w)
    print(out)
