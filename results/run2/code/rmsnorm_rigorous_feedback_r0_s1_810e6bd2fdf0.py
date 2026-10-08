import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, N, M, BLOCK_SIZE: tl.constexpr):
    # Define the position of each element within the block
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    block_start = pid * BLOCK_SIZE
    rows = tl.arange(0, BLOCK_SIZE)
    mask = rows < N - block_start
    x = tl.load(x_ptr + block_start + rows[:, None], mask=mask[:, None])
    
    # Compute the mean of squares
    sum_squares = tl.sum(x * x, axis=0)
    mean_squares = sum_squares / N
    
    # Compute the root mean square and add epsilon
    rms = tl.sqrt(mean_squares + 1e-6)
    
    # Normalize and scale by weight
    w = tl.load(w_ptr)
    normalized_x = x / rms
    scaled_x = normalized_x * w
    
    # Store the result
    tl.store(out_ptr + block_start + rows[:, None], scaled_x, mask=mask[:, None])

def kernel(x, w):
    N, M = x.shape
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE), M)
    out = torch.empty_like(x)
    rmsnorm_kernel[grid](x, w, out, N, M, BLOCK_SIZE)
    return out

# Example usage
if __name__ == "__main__":
    x = torch.randn((1024, 512), device='cuda', dtype=torch.float16)
    w = torch.randn((512,), device='cuda', dtype=torch.float16)
    result = kernel(x, w)
    print(result)
