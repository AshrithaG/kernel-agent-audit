import torch
import triton
import triton.language as tl
import math

# Triton kernel for adding residual and performing RMSNorm
@triton.jit
def add_rmsnorm_kernel(x_ptr, r_ptr, w_ptr, out_ptr, stride_x, stride_r, stride_w, stride_out, N, H, BLOCK_SIZE: tl.constexpr):
    # Define the indices for the threads
    idx = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)

    # Calculate the start index for the current block
    start_idx = idx * BLOCK_SIZE

    # Load the input data into registers
    x = tl.load(x_ptr + start_idx, mask=start_idx < N, other=0.0)
    r = tl.load(r_ptr + start_idx, mask=start_idx < N, other=0.0)
    w = tl.load(w_ptr + pid, mask=pid < H, other=0.0)

    # Add the residual
    h = x + r

    # Compute the mean of squares
    sum_squares = tl.sum(h * h, axis=0)

    # Compute the RMSNorm
    rms = tl.sqrt(sum_squares / N + 1e-6)

    # Normalize and scale
    normalized_h = h / rms
    scaled_h = normalized_h * w

    # Store the result
    tl.store(out_ptr + start_idx, scaled_h, mask=start_idx < N)

# Triton wrapper function
def add_rmsnorm_triton(x, r, w):
    # Ensure the inputs are float16
    x = x.float16()
    r = r.float16()
    w = w.float16()

    # Get the shapes
    B, H = x.shape[:-1]
    N = x.shape[-1]

    # Allocate output tensor
    out = torch.empty_like(x)

    # Define the grid and block sizes
    BLOCK_SIZE = 128
    grid = lambda opt: (triton.cdiv(N, BLOCK_SIZE), H)

    # Launch the Triton kernel
    add_rmsnorm_kernel[grid](x, r, w, out, x.stride(-1), r.stride(-1), w.stride(0), out.stride(-1), N, H, BLOCK_SIZE=BLOCK_SIZE)

    # Return the output tensor
    return out

# Reference implementation for comparison
def add_rmsnorm_reference(x, r, w):
    h = x.float() + r.float()
    return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)

# Example usage
if __name__ == "__main__":
    # Create example tensors
    B, H, N = 32, 128, 512
    x = torch.randn(B, H, N, device="cuda", dtype=torch.float16)
    r = torch.randn(B, H, N, device="cuda", dtype=torch.float16)
    w = torch.randn(H, device="cuda", dtype=torch.float16)

    # Run the Triton implementation
    out_triton = add_rmsnorm_triton(x, r, w)

    # Run the reference implementation
    out_ref = add_rmsnorm_reference(x, r, w)

    # Check if the outputs are the same
    assert torch.allclose(out_triton, out_ref, atol=1e-3, rtol=1e-3), "Outputs do not match"
    print("Outputs match within tolerance")
