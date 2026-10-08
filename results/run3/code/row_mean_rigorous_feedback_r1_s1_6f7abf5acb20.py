import torch
import triton
import triton.language as tl
import time

# Improved kernel
@triton.jit
def _row_mean(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)

    # Vectorized reduction
    sum_val = tl.sum(x, axis=0)
    mean = sum_val / n_cols
    tl.store(out_ptr + row, mean.to(out_ptr.dtype.element_ty), mask=True)

def kernel(x):
    x = x.contiguous()
    out = torch.empty((x.shape[0],), dtype=torch.float16, device=x.device)
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    _row_mean[(n_rows,)](x, out, n_cols, BLOCK=BLOCK)
    return out

# Reference implementation
def row_mean(x):
    return x.float().mean(dim=-1).to(x.dtype)

# Timing function
def time_kernel(kernel, x):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    y = kernel(x)
    end_event.record()
    torch.cuda.synchronize()
    
    return start_event.elapsed_time(end_event)

# Test shapes
shapes = [
    (1024, 1024),
    (4096, 4096),
    (1000, 3001),
    (1, 50257),
    (8192, 128)
]

# Measure performance
for shape in shapes:
    x = torch.randn(*shape, dtype=torch.float16, device='cuda')
    
    eager_time = time_kernel(row_mean, x)
    triton_time = time_kernel(kernel, x)
    
    print(f"Shape {shape}: {eager_time / triton_time:.2f}x eager PyTorch, {torch.compile(row_mean)(x).elapsed_time(torch.compile(kernel)(x)) / triton_time:.2f}x torch.compile")
