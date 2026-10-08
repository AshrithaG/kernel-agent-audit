import torch
import triton
import triton.language as tl

@triton.jit
def mean_kernel(x_ptr, y_ptr, stride_x, N, BLOCK_SIZE: tl.constexpr):
    # Load data into shared memory
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    row_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_idx = bid
    mask = row_idx < N
    x = tl.load(x_ptr + row_idx[:, None] * stride_x + col_idx[None, :], mask=mask[:, None], other=0.0)
    
    # Compute sum and count
    sum_val = tl.sum(x, axis=0)
    count = tl.sum(mask, axis=0)
    
    # Store result
    if tl.is_first_block(column=0):
        y = sum_val / count
        tl.store(y_ptr + col_idx, y)

def kernel(x):
    M, N = x.shape
    y = torch.empty(M, dtype=x.dtype, device=x.device)
    
    # Define grid and block dimensions
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE), M)
    
    # Launch the kernel
    mean_kernel[grid](x.contiguous(), y.contiguous(), x.stride(1), N, BLOCK_SIZE)
    
    return y

# Example usage
if __name__ == "__main__":
    import torch
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    y_triton = kernel(x)
    y_ref = x.float().mean(dim=-1).to(x.dtype)
    assert torch.allclose(y_triton, y_ref, atol=1e-3, rtol=1e-3)
    print("Triton kernel output matches reference implementation.")
