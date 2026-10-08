import torch
import triton
import triton.language as tl
import math

# Triton kernel for LayerNorm
@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, D,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    EPS: tl.constexpr
):
    # Define the position of the element to work on
    row = tl.program_id(0) * BLOCK_SIZE_N + tl threadIdx(0)
    col = tl.program_id(1) * BLOCK_SIZE_D + tl threadIdx(1)

    # Load the data into shared memory
    x_shared = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_D), dtype=tl.float32)
    if row < N and col < D:
        x_shared[tl.threadIdx(0), tl.threadIdx(1)] = tl.load(x_ptr + row * D + col, mask=row < N & col < D)

    # Compute the mean
    mean = tl.sum(x_shared, axis=1) / D
    mean = tl.broadcast_to(mean[:, None], (BLOCK_SIZE_N, BLOCK_SIZE_D))

    # Compute the variance
    var = tl.sum((x_shared - mean) ** 2, axis=1) / D
    var = tl.broadcast_to(var[:, None], (BLOCK_SIZE_N, BLOCK_SIZE_D))

    # Compute the normalized value
    x_normalized = (x_shared - mean) / tl.sqrt(var + EPS)

    # Scale and shift by the weights and biases
    w = tl.load(w_ptr + col, mask=col < D)
    b = tl.load(b_ptr + col, mask=col < D)
    out = x_normalized * w + b

    # Store the result back to global memory
    if row < N and col < D:
        tl.store(out_ptr + row * D + col, out, mask=row < N & col < D)

# Wrapper function to call the Triton kernel
def kernel(x, w, b, eps=1e-5):
    N, D = x.shape
    assert x.is_cuda and w.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == torch.float16 and w.dtype == torch.float16 and b.dtype == torch.float16, "Inputs must be float16"

    # Allocate output tensor
    out = torch.empty_like(x)

    # Define grid and block sizes
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_D = 32
    grid = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(D, BLOCK_SIZE_D))

    # Launch the kernel
    layernorm_kernel[grid](
        x.data_ptr(), w.data_ptr(), b.data_ptr(), out.data_ptr(),
        N, D,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        EPS=eps
    )

    return out

# Example usage
if __name__ == "__main__":
    import torch.nn.functional as F

    # Create random input tensors
    torch.manual_seed(0)
    x = torch.randn(128, 512, device='cuda', dtype=torch.float16)
    w = torch.randn(512, device='cuda', dtype=torch.float16)
    b = torch.randn(512, device='cuda', dtype=torch.float16)

    # Reference implementation
    ref_out = F.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float(), 1e-5).to(x.dtype)

    # Triton implementation
    triton_out = kernel(x, w, b)

    # Check correctness
    assert torch.allclose(ref_out, triton_out, atol=1e-3, rtol=1e-3)

    print("Reference output:", ref_out)
    print("Triton output:", triton_out)
