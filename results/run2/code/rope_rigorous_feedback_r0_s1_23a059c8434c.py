import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel(
    X, C, S, 
    stride_x_tokens, stride_x_heads, stride_x_dim,
    stride_c_tokens, stride_c_dim,
    stride_s_tokens, stride_s_dim,
    N, H, D
):
    pid = tl.program_id(axis=0)
    BLOCK_SIZE = tl.constexpr(BLOCK_SIZE)
    block_id = pid // (H * BLOCK_SIZE)
    head_id = (pid // BLOCK_SIZE) % H
    element_id = pid % BLOCK_SIZE

    off_x = block_id * BLOCK_SIZE + element_id
    if off_x >= N:
        return

    # Load x1 and x2
    off_x1 = off_x * stride_x_tokens + head_id * stride_x_heads
    off_x2 = off_x1 + D // 2
    x1 = tl.load(X + off_x1, mask=off_x < N, other=0.0)
    x2 = tl.load(X + off_x2, mask=off_x < N, other=0.0)

    # Load cos and sin
    off_c = off_x * stride_c_tokens
    off_s = off_x * stride_s_tokens
    c = tl.load(C + off_c, mask=off_x < N, other=0.0)
    s = tl.load(S + off_s, mask=off_x < N, other=0.0)

    # Compute the rotated values
    x1_rotated = x1 * c - x2 * s
    x2_rotated = x2 * c + x1 * s

    # Store the results back to X
    off_out1 = off_x1
    off_out2 = off_x2
    tl.store(X + off_out1, x1_rotated, mask=off_x < N)
    tl.store(X + off_out2, x2_rotated, mask=off_x < N)

def kernel(X, C, S):
    N, H, D = X.shape
    BLOCK_SIZE = 128
    grid = lambda opt: (triton.cdiv(N, BLOCK_SIZE) * H, )
    triton.rotary_position_embedding_kernel[grid](
        X, C, S,
        X.stride(0), X.stride(1), X.stride(2),
        C.stride(0), C.stride(1),
        S.stride(0), S.stride(1),
        N, H, D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return X

# Example usage
if __name__ == "__main__":
    import torch
    tokens, heads, dim = 1024, 8, 128
    x = torch.randn(tokens, heads, dim, device='cuda', dtype=torch.float16)
    cos = torch.randn(tokens, dim // 2, device='cuda', dtype=torch.float16)
    sin = torch.randn(tokens, dim // 2, device='cuda', dtype=torch.float16)

    x_ref = x.clone()
    reference_output = rope(x_ref, cos, sin)

    x_triton = x.clone()
    triton_output = kernel(x_triton, cos, sin)

    print("Reference output:", reference_output)
    print("Triton output:", triton_output)
    print("Are outputs equal?", torch.allclose(reference_output, triton_output))
