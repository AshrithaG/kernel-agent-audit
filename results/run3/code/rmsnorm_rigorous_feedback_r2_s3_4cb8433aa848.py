import torch
import triton
import triton.language as tl
import time

# Define the improved kernel
@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute the mean of squares using warp-reduction
    sum_squares = tl.sum(x * x, axis=0)
    mean_squares = sum_squares / n_cols
    
    # Compute the rsqrt
    rsqrt_mean_squares = tl.rsqrt(mean_squares + 1e-6)
    
    # Normalize and scale
    normalized_x = x * rsqrt_mean_squares
    scaled_x = normalized_x * w
    
    tl.store(out_ptr + row * n_cols + cols, scaled_x.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, w):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    _rmsnorm_kernel[(n_rows,)](x, w, out, n_cols, BLOCK=triton.next_power_of_2(n_cols))
    return out

# Test the improved kernel
shapes = [
    (1024, 1024),
    (4096, 4096),
    (1000, 3001),
    (1, 8192),
    (8192, 128)
]

for shape in shapes:
    x = torch.randn(shape, device='cuda', dtype=torch.float16)
    w = torch.randn(shape[1], device='cuda', dtype=torch.float16)
    
    # Warm-up
    for _ in range(3):
        kernel(x, w)
    
    # Measure time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(10):
        kernel(x, w)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    
    print(f"Shape {shape}: {elapsed_time_ms:.2f} ms")
