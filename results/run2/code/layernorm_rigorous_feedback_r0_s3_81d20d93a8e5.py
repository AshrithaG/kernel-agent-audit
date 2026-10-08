import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    X, W, B, Y,
    N, D,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    indices = tl.arange(0, BLOCK_SIZE)
    mask = indices < D

    x = tl.load(X + block_start, mask=mask, other=0.0)
    
    mean = tl.sum(x, axis=1) / D
    mean = tl.broadcast_to(mean[:, None], (BLOCK_SIZE, D))
    
    variance = tl.sum((x - mean) ** 2, axis=1) / D
    variance = tl.broadcast_to(variance[:, None], (BLOCK_SIZE, D))
    
    inv_var = 1.0 / tl.sqrt(variance + EPS)
    
    normalized_x = (x - mean) * inv_var
    
    y = normalized_x * W + B
    
    tl.store(Y + block_start, y, mask=mask)

def kernel(X, W, B, Y):
    N, D = X.shape
    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
    
    layer_norm_kernel[grid](X, W, B, Y, N, D, BLOCK_SIZE=BLOCK_SIZE, EPS=1e-5)

# Example usage:
if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(32, 512, device='cuda', dtype=torch.float16)
    W = torch.randn(512, device='cuda', dtype=torch.float16)
    B = torch.randn(512, device='cuda', dtype=torch.float16)
    Y = torch.empty_like(X)

    kernel(X, W, B, Y)

    print(Y)
