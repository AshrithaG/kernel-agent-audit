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
    mean = tl.sum(mean) / (N * BLOCK_SIZE_D)
    mean = tl.full((BLOCK_SIZE_N, BLOCK_SIZE_D), mean, dtype=tl.float32)

    # Compute the variance
    var = tl.sum((x_shared - mean) ** 2, axis=1) / D
    var = tl.sum(var) / (N * BLOCK_SIZE_D)
    var = tl.full((BLOCK_SIZE_N, BLOCK_SIZE_D), var, dtype=tl.float32)

    # Compute the output
    inv_var = tl.sqrt(var + EPS)
    normalized = (x_shared - mean) / inv_var
    weighted = normalized * tl.load(w_ptr + col, mask=col < D)
    biased = weighted + tl.load(b_ptr + col, mask=col < D)

    # Store the result
    if row < N and col < D:
        tl.store(out_ptr + row * D + col, biased, mask=row < N & col < D)

# Triton wrapper function
def layernorm_triton(x, w, b, eps=1e-5):
    N, D = x.shape
    assert x.dtype == torch.float16, "Input tensor must be of type float16"
    assert w.dtype == torch.float16, "Weight tensor must be of type float16"
    assert b.dtype == torch.float16, "Bias tensor must be of type float16"

    # Allocate output tensor
    out = torch.empty_like(x)

    # Define grid and block sizes
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_D = 32
    grid = (
        triton.cdiv(N, BLOCK_SIZE_N),
        triton.cdiv(D, BLOCK_SIZE_D),
    )

    # Launch the kernel
    layernorm_kernel[grid](
        x.contiguous().cuda(),
        w.contiguous().cuda(),
        b.contiguous().cuda(),
        out.contiguous().cuda(),
        N, D,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        EPS=eps,
    )

    return out

# Reference implementation for comparison
def layernorm_reference(x, w, b, eps=1e-5):
    return torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float(), eps).to(x.dtype)

# Example usage
if __name__ == "__main__":
    # Create random input tensors
    torch.manual_seed(0)
    N, D = 1024, 512
    x = torch.randn(N, D, device='cuda', dtype=torch.float16)
    w = torch.randn(D, device='cuda', dtype=torch.float16)
    b = torch.randn(D, device='cuda', dtype=torch.float16)

    # Run the Triton implementation
    out_triton = layernorm_triton(x, w, b)

    # Run the reference implementation
    out_reference = layernorm_reference(x, w, b)

    # Check the results
    print("Triton output:", out_triton)
    print("Reference output:", out_reference)
    print("Max absolute difference:", torch.max(torch.abs(out_triton - out_reference)))
