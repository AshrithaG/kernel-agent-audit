import torch
import triton
import triton.language as tl

@triton.jit
def swiglu_kernel(x_ptr, out_ptr, N, H, BLOCK_SIZE: tl.constexpr):
    # Define the indices for the current thread
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    # Calculate the global index for the current thread
    row_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_idx = bid * H
    # Load the data into registers
    x = tl.load(x_ptr + row_idx[:, None] * H + col_idx, mask=row_idx[:, None] < N, other=0.0)
    # Split the data into two halves
    x1 = x[:, :H//2]
    x2 = x[:, H//2:]
    # Apply SiLU activation to the first half
    x1_silu = x1 * tl.sigmoid(x1)
    # Multiply the SiLU result with the second half
    out = x1_silu * x2
    # Store the result back to global memory
    tl.store(out_ptr + row_idx[:, None] * H + col_idx, out, mask=row_idx[:, None] < N)

def kernel(*inputs):
    # Ensure the input is a float16 tensor
    x = inputs[0].contiguous().to(torch.float16)
    N, H = x.shape
    # Determine the block size
    BLOCK_SIZE = 128
    # Allocate output buffer
    out = torch.empty_like(x)
    # Launch the Triton kernel
    grid = lambda opt: (triton.cdiv(N, BLOCK_SIZE), H // (BLOCK_SIZE // 32))
    swiglu_kernel[grid](x.data_ptr(), out.data_ptr(), N, H, BLOCK_SIZE=BLOCK_SIZE)
    return out

# Example usage:
if __name__ == "__main__":
    import torch.nn.functional as F
    import math

    # Create a random float16 tensor
    x = torch.randn(1024, 256).cuda().half()
    # Reference implementation
    ref_output = (F.silu(x[..., :128].float()) * x[..., 128:].float()).to(x.dtype)
    # Triton kernel implementation
    triton_output = kernel(x)
    # Check if the outputs are close
    assert torch.allclose(ref_output, triton_output, atol=1e-3, rtol=1e-3)
    print("Outputs match within tolerance!")
