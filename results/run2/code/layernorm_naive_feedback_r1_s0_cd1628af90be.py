import torch
import triton
import triton.language as tl

# Triton kernel for layer normalization
@triton.jit
def layer_norm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, mean_ptr, rsqrt_var_ptr, N, D, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < D

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / N
    tl.store(mean_ptr, mean)

    # Compute variance
    var = tl.sum((x - mean) ** 2, axis=0) / N
    rsqrt_var = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(rsqrt_var_ptr, rsqrt_var)

    # Normalize and scale
    y = (x - mean) * rsqrt_var
    y = y * tl.load(w_ptr + offsets, mask=mask, other=0.0) + tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, y, mask=mask)

# Triton wrapper function
def kernel(x, w, b):
    N, D = x.shape
    BLOCK_SIZE = 128
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
    
    mean = torch.empty((N,), device=x.device, dtype=torch.float32)
    rsqrt_var = torch.empty((N,), device=x.device, dtype=torch.float32)
    
    layer_norm_kernel[grid](x, w, b, x, mean, rsqrt_var, N, D, BLOCK_SIZE=BLOCK_SIZE)
    
    return x

# Reference implementation for verification
def reference(x, w, b):
    return torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float(), 1e-5).to(x.dtype)

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(128, 512, device='cuda', dtype=torch.float16)
    w = torch.randn(512, device='cuda', dtype=torch.float16)
    b = torch.randn(512, device='cuda', dtype=torch.float16)

    output_triton = kernel(x.clone(), w.clone(), b.clone())
    output_reference = reference(x.clone(), w.clone(), b.clone())

    print("Output matches:", torch.allclose(output_triton, output_reference))
