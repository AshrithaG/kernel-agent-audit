import torch
import triton
import triton.language as tl

# Triton kernel for rotary position embedding
@triton.jit
def rope_kernel(
    X, C, S, 
    stride_x_tokens, stride_x_heads, stride_x_dim,
    stride_c_tokens, stride_c_dim,
    stride_s_tokens, stride_s_dim,
    N, H, D
):
    # Define the index of the current thread
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Check if the current thread is out of bounds
    if token_idx >= N or head_idx >= H:
        return
    
    # Load x1 and x2 from X
    x1_ptr = X + token_idx * stride_x_tokens + head_idx * stride_x_heads
    x2_ptr = x1_ptr + stride_x_dim // 2
    x1 = tl.load(x1_ptr, mask=token_idx < N, other=0.0)
    x2 = tl.load(x2_ptr, mask=token_idx < N, other=0.0)
    
    # Load cos and sin from C and S
    c_ptr = C + token_idx * stride_c_tokens
    s_ptr = S + token_idx * stride_s_tokens
    c = tl.load(c_ptr, mask=token_idx < N, other=0.0)
    s = tl.load(s_ptr, mask=token_idx < N, other=0.0)
    
    # Compute the rotated embeddings
    x1_rot = x1 * c - x2 * s
    x2_rot = x2 * c + x1 * s
    
    # Store the results back to X
    tl.store(x1_ptr, x1_rot, mask=token_idx < N)
    tl.store(x2_ptr, x2_rot, mask=token_idx < N)

def kernel(X, C, S):
    # Get the shapes of the input tensors
    N, H, D = X.shape
    _, D_half = C.shape
    
    # Ensure the dimensions are compatible
    assert D == 2 * D_half, "Dimension mismatch between X and C/S"
    
    # Define the grid of threads
    grid = (triton.cdiv(N, 128), H)
    
    # Launch the Triton kernel
    rope_kernel[grid](
        X, C, S,
        stride_x_tokens=X.stride(0),
        stride_x_heads=X.stride(1),
        stride_x_dim=X.stride(2),
        stride_c_tokens=C.stride(0),
        stride_c_dim=C.stride(1),
        stride_s_tokens=S.stride(0),
        stride_s_dim=S.stride(1),
        N=N, H=H, D=D
    )
    
    return X

# Example usage
if __name__ == "__main__":
    # Create example inputs
    tokens, heads, dim = 1024, 8, 64
    X = torch.randn(tokens, heads, dim, device='cuda', dtype=torch.float16)
    cos = torch.randn(tokens, dim // 2, device='cuda', dtype=torch.float16)
    sin = torch.randn(tokens, dim // 2, device='cuda', dtype=torch.float16)
    
    # Apply the kernel
    result = kernel(X.clone(), cos.clone(), sin.clone())
    
    # Verify the result
    ref_result = rope(X.clone(), cos.clone(), sin.clone())
    assert torch.allclose(result, ref_result, atol=1e-3), "The Triton kernel output does not match the reference implementation."
    
    print("Kernel executed successfully and verified.")
