import torch
import triton
import triton.language as tl

@triton.jit
def rotary_position_embedding_kernel(
    X, C, S, 
    X_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    M = tl.numel(X) // (BLOCK_M * BLOCK_N)
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    off_m = m * BLOCK_M
    off_n = n * BLOCK_N
    
    X_ptrs = X + off_m * BLOCK_N + off_n
    C_ptrs = C + off_m * BLOCK_N + off_n
    S_ptrs = S + off_m * BLOCK_N + off_n
    X_out_ptrs = X_out + off_m * BLOCK_N + off_n
    
    mask_m = off_m < M
    mask_n = off_n < BLOCK_N
    
    d = BLOCK_N // 2
    
    for k in range(0, d, BLOCK_N):
        if mask_m & mask_n:
            x1 = tl.load(X_ptrs + k, mask=mask_n, other=0.0)
            x2 = tl.load(X_ptrs + k + d, mask=mask_n, other=0.0)
            c = tl.load(C_ptrs + k, mask=mask_n, other=0.0)
            s = tl.load(S_ptrs + k, mask=mask_n, other=0.0)
            
            x1_out = x1 * c - x2 * s
            x2_out = x2 * c + x1 * s
            
            tl.store(X_out_ptrs + k, x1_out, mask=mask_n)
            tl.store(X_out_ptrs + k + d, x2_out, mask=mask_n)

def kernel(X, C, S):
    assert X.dtype == torch.float16
    assert C.dtype == torch.float16
    assert S.dtype == torch.float16
    assert X.shape[0] == C.shape[0]
    assert X.shape[0] == S.shape[0]
    assert X.shape[-1] % 2 == 0
    assert X.shape[-1] // 2 == C.shape[-1] == S.shape[-1]
    
    M, H, N = X.shape
    d = N // 2
    
    X_out = torch.empty_like(X)
    
    grid = lambda META: (
        triton.cdiv(M * H, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )
    
    rotary_position_embedding_kernel[grid](
        X, C, S, X_out,
        BLOCK_M=128,
        BLOCK_N=N,
    )
    
    return X_out

# Example usage:
# X = torch.randn(1024, 8, 128, dtype=torch.float16, device='cuda')
# C = torch.randn(1024, 64, dtype=torch.float16, device='cuda')
# S = torch.randn(1024, 64, dtype=torch.float16, device='cuda')
# result = kernel(X, C, S)
