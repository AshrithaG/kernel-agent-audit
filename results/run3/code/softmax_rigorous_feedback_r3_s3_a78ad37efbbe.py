import torch
import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=float('-inf')).to(tl.float32)
    
    # Compute max for numerical stability
    max_x = tl.max(x, axis=0)
    x = x - max_x
    
    # Compute exp
    exp_x = tl.exp(x)
    
    # Compute sum of exps
    sum_exp_x = tl.sum(exp_x, axis=0)
    
    # Compute softmax
    softmax_x = exp_x / sum_exp_x
    
    tl.store(out_ptr + row * n_cols + cols, softmax_x.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    _softmax_kernel[grid](x, out, n_cols, BLOCK=BLOCK_SIZE)
    return out

# Example usage:
if __name__ == "__main__":
    import time
    import torch

    def measure_time(func, *args, num_iterations=10):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_iterations):
            func(*args)
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / num_iterations

    shapes = [
        (1024, 1024),
        (4096, 4096),
        (1000, 3001),
        (1, 50257),
        (8192, 128)
    ]

    for shape in shapes:
        x = torch.randn(shape, device='cuda', dtype=torch.float16)
        
        # Eager PyTorch
        torch.cuda.synchronize()
        start_time = time.time()
        torch.softmax(x.float(), dim=-1).to(x.dtype)
        torch.cuda.synchronize()
        eager_time = time.time() - start_time
        
        # Triton kernel
        torch.cuda.synchronize()
        start_time = time.time()
        kernel(x)
        torch.cuda.synchronize()
        triton_time = time.time() - start_time
        
        # Torch.compile
        torch.cuda.synchronize()
        compiled_func = torch.compile(kernel)
        start_time = time.time()
        compiled_func(x)
        torch.cuda.synchronize()
        torch_compile_time = time.time() - start_time
        
        print(f"Shape {shape}:")
        print(f"  Eager PyTorch: {eager_time:.4f}s")
        print(f"  Triton Kernel: {triton_time:.4f}s")
        print(f"  Torch.compile: {torch_compile_time:.4f}s")
        print()
