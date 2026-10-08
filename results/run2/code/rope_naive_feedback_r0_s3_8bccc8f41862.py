import torch
import triton
import triton.language as tl

# Triton kernel for rotary position embedding
@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, output_ptr,
    tokens, heads, dim,
    BLOCK_SIZE: tl.constexpr
):
    # Indices
    row = tl.program_id(0)  # token index
    col = tl.program_id(1)  # head index
    idx = tl.arange(0, BLOCK_SIZE)

    # Load data
    x1 = tl.load(x_ptr + row * heads * dim + col * dim + idx, mask=(idx < dim // 2), other=0.0)
    x2 = tl.load(x_ptr + row * heads * dim + col * dim + idx + dim // 2, mask=(idx < dim // 2), other=0.0)
    c = tl.load(cos_ptr + row * dim // 2 + idx, mask=(idx < dim // 2), other=0.0)
    s = tl.load(sin_ptr + row * dim // 2 + idx, mask=(idx < dim // 2), other=0.0)

    # Compute rotated embeddings
    x1_rot = x1 * c - x2 * s
    x2_rot = x2 * c + x1 * s

    # Store results
    tl.store(output_ptr + row * heads * dim + col * dim + idx, x1_rot, mask=(idx < dim // 2))
    tl.store(output_ptr + row * heads * dim + col * dim + idx + dim // 2, x2_rot, mask=(idx < dim // 2))

# Wrapper function to launch the Triton kernel
def kernel(x, cos, sin):
    tokens, heads, dim = x.shape
    assert dim % 2 == 0, "Dimension must be even"
    assert cos.shape == (tokens, dim // 2)
    assert sin.shape == (tokens, dim // 2)

    # Allocate output tensor
    output = torch.empty_like(x)

    # Launch Triton kernel
    grid = (tokens, heads)
    BLOCK_SIZE = 32
    rope_kernel[grid](
        x.contiguous().data_ptr(),
        cos.contiguous().data_ptr(),
        sin.contiguous().data_ptr(),
        output.data_ptr(),
        tokens, heads, dim,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output

# Example usage
if __name__ == "__main__":
    import math

    # Create dummy input tensors
    tokens, heads, dim = 16, 8, 64
    x = torch.randn(tokens, heads, dim, dtype=torch.float16, device='cuda')
    cos = torch.cos(torch.arange(tokens * dim // 2, dtype=torch.float16, device='cuda').view(tokens, dim // 2) / dim)
    sin = torch.sin(torch.arange(tokens * dim // 2, dtype=torch.float16, device='cuda').view(tokens, dim // 2) / dim)

    # Compute reference output
    reference_output = rope(x, cos, sin)

    # Compute Triton output
    triton_output = kernel(x, cos, sin)

    # Check if outputs are equal
    assert torch.allclose(reference_output, triton_output, atol=1e-3, rtol=1e-3), "Outputs do not match"

    print("Triton kernel works correctly!")
