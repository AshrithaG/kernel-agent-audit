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

    # Compute the rotated positions
    x1_rot = x1 * cos_val - x2 * sin_val
    x2_rot = x2 * cos_val + x1 * sin_val

    # Store the results
    tl.store(out_ptr + x_offset, x1_rot, mask=token_idx < tokens)
    tl.store(out_ptr + x_offset + d, x2_rot, mask=token_idx < tokens)

# Define the wrapper function
def kernel(x, cos, sin):
    tokens, heads, dim = x.shape
    assert dim % 2 == 0, "Dimension must be even"
    assert cos.shape == (tokens, dim // 2)
    assert sin.shape == (tokens, dim // 2)

    # Allocate output tensor
    out = torch.empty_like(x)

    # Define grid
    BLOCK_SIZE = 128
    grid = (triton.cdiv(tokens, BLOCK_SIZE), heads)

    # Launch the kernel
    rope_kernel[grid](x, cos, sin, out, tokens, heads, dim, BLOCK_SIZE)

    return out

# Example usage
if __name__ == "__main__":
    import math

    # Create example input tensors
    tokens = 16
    heads = 8
    dim = 32
    x = torch.rand((tokens, heads, dim), device="cuda", dtype=torch.float16)
    cos = torch.cos(torch.arange(tokens * dim // 2, device="cuda", dtype=torch.float16).view(tokens, dim // 2))
    sin = torch.sin(torch.arange(tokens * dim // 2, device="cuda", dtype=torch.float16).view(tokens, dim // 2))

    # Run the kernel
    out = kernel(x, cos, sin)

    # Verify correctness
    ref_out = torch.cat([x[..., :dim//2] * cos[:, None, :] - x[..., dim//2:] * sin[:, None, :],
                         x[..., dim//2:] * cos[:, None, :] + x[..., :dim//2] * sin[:, None, :]], dim=-1)

    print("Output matches reference:", torch.allclose(out, ref_out))
