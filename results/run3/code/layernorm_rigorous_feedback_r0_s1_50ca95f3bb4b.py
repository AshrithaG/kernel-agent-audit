import torch
import triton
import triton.language as tl

@triton.jit
def _layernorm(x_ptr, w_ptr, b_ptr, out_ptr, n_elements, eps: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_elements
    
    # Load data
    x = tl.load(x_ptr + row * n_elements + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean
    mean = tl.sum(x, axis=1) / n_elements
    mean = tl.broadcast_to(mean[:, None], (BLOCK,))
    
    # Compute variance
    var = tl.sum((x - mean) ** 2, axis=1) / n_elements
    var = tl.broadcast_to(var[:, None], (BLOCK,))
    
    # Compute normalized values
    epsilon = tl.full((BLOCK,), eps, dtype=tl.float32)
    rstd = tl.sqrt(var + epsilon)
    normalized = (x - mean) / rstd
    
    # Load weights and biases
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Apply weights and biases
    output = normalized * w + b
    
    # Store results
    tl.store(out_ptr + row * n_elements + cols, output.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, w, b):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_elements = x.shape
    eps = 1e-5
    BLOCK = triton.next_power_of_2(n_elements)
    _layernorm[(n_rows,)](x, w, b, out, n_elements, eps, BLOCK=BLOCK)
    return out
