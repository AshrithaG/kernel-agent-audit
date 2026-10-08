import torch
import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    tokens, heads, dim,
    BLOCK_SIZE: tl.constexpr
):
    # Each thread handles one token and one head
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if token_idx >= tokens or head_idx >= heads:
        return

    # Load the data
    d = dim // 2
    x_offset = (token_idx * heads + head_idx) * dim
    x1 = tl.load(x_ptr + x_offset, mask=token_idx < tokens, other=0.0)
    x2 = tl.load(x_ptr + x_offset + d, mask=token_idx < tokens, other=0.0)

    cos_offset = token_idx * d
    cos_val = tl.load(cos_ptr + cos_offset, mask=token_idx < tokens, other=0.0)
    sin_val = tl.load(sin_ptr + cos_offset, mask=token_idx < tokens, other=0.0)

    # Apply the rotation
    x1_rotated = x1 * cos_val - x2 * sin_val
    x2_rotated = x2 * cos_val + x1 * sin_val

    # Store the result
    out_offset = (token_idx * heads + head_idx) * dim
    tl.store(out_ptr + out_offset, x1_rotated, mask=token_idx < tokens)
    tl.store(out_ptr + out_offset + d, x2_rotated, mask=token_idx < tokens)

# Define the wrapper function
def kernel(x, cos, sin):
    tokens, heads, dim = x.shape
    assert dim % 2 == 0, "Dimension must be even"
    assert cos.shape == (tokens, dim // 2)
    assert sin.shape == (tokens, dim // 2)

    out = torch.empty_like(x, dtype=torch.float32)

    # Launch the Triton kernel
    grid = (
        triton.cdiv(tokens, 16),
        heads,
        1
    )
    BLOCK_SIZE = 16
    rope_kernel[grid](
        x.contiguous().view(-1).data_ptr(),
        cos.contiguous().view(-1).data_ptr(),
        sin.contiguous().view(-1).data_ptr(),
        out.contiguous().view(-1).data_ptr(),
        tokens, heads, dim,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out.to(x.dtype)

# Example usage
if __name__ == "__main__":
    import time

    # Create example inputs
    tokens = 1024
    heads = 8
    dim = 128
    x = torch.rand((tokens, heads, dim), device='cuda', dtype=torch.float16)
    cos = torch.rand((tokens, dim // 2), device='cuda', dtype=torch.float16)
    sin = torch.rand((tokens, dim // 2), device='cuda', dtype=torch.float16)

    # Measure the time taken by the reference implementation
    start_time = time.time()
    ref_out = rope(x, cos, sin)
    ref_time = time.time() - start_time
    print(f"Reference implementation time: {ref_time:.6f} seconds")

    # Measure the time taken by the Triton kernel
    start_time = time.time()
    triton_out = kernel(x, cos, sin)
    triton_time = time.time() - start_time
    print(f"Triton kernel time: {triton_time:.6f} seconds")

    # Check the outputs are the same
    assert torch.allclose(ref_out, triton_out, atol=1e-5, rtol=1e-5), "Outputs do not match"
    print("Outputs match within tolerance")
