import torch
import triton
import triton.language as tl

# Triton kernel for RMSNorm
@triton.jit
def rmsnorm_kernel(
    x_ptr, w_ptr, out_ptr,
    stride_x, stride_w, stride_out,
    N, M,
    BLOCK_M: tl.constexpr
):
    # Define the block index
    pid = tl.program_id(axis=0)
    # Calculate the row index for this block
    row = pid * BLOCK_M
    # Load the data into registers
    rows = tl.arange(0, BLOCK_M)
    mask = rows < N
    x = tl.load(x_ptr + row * stride_x + rows[:, None], mask=mask, other=0.0)
    w = tl.load(w_ptr + rows, mask=mask, other=0.0)
    # Compute the square root of the mean of squares
    sum_squares = tl.sum(x * x, axis=1)
    rsqrt_sum_squares = 1.0 / tl.sqrt(sum_squares + 1e-6)
    # Normalize and scale
    out = x * rsqrt_sum_squares[:, None] * w
    # Store the result
    tl.store(out_ptr + row * stride_out + rows[:, None], out, mask=mask)

# Triton wrapper function
def kernel(x, w):
    assert x.is_cuda and w.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == torch.float16 and w.dtype == torch.float16, "Inputs must be float16"
    assert x.ndim == 2 and w.ndim == 1 and x.shape[1] == w.shape[0], "Invalid input shapes"

    N, M = x.shape
    BLOCK_M = 128

    # Allocate output tensor
    out = torch.empty_like(x)

    # Launch the Triton kernel
    grid = lambda META: (triton.cdiv(N, BLOCK_M),)
    rmsnorm_kernel[grid](
        x_ptr=x.data_ptr(),
        w_ptr=w.data_ptr(),
        out_ptr=out.data_ptr(),
        stride_x=x.stride(0),
        stride_w=w.stride(0),
        stride_out=out.stride(0),
        N=N, M=M,
        BLOCK_M=BLOCK_M,
    )

    return out

# Example usage
if __name__ == "__main__":
    import time

    # Create random input tensors
    torch.manual_seed(0)
    x = torch.randn(512, 1024, device='cuda', dtype=torch.float16)
    w = torch.randn(1024, device='cuda', dtype=torch.float16)

    # Measure the time taken by the Triton kernel
    start_time = time.time()
    out_triton = kernel(x, w)
    end_time = time.time()
    print(f"Triton kernel execution time: {end_time - start_time:.4f} seconds")

    # Measure the time taken by the reference implementation
    start_time = time.time()
    out_reference = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w).to(x.dtype)
    end_time = time.time()
    print(f"Reference implementation execution time: {end_time - start_time:.4f} seconds")

    # Check if the outputs are the same
    assert torch.allclose(out_triton, out_reference, atol=1e-3, rtol=1e-3), "Outputs do not match"
    print("Outputs match within tolerance")
