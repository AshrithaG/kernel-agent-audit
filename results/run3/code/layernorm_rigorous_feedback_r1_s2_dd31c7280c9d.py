import torch
import triton
import triton.language as tl

@triton.jit
def _layernorm(x_ptr, w_ptr, b_ptr, out_ptr, n_features, BLOCK: tl.constexpr):
    # Define the block size and the number of elements per block
    BLOCK_SIZE = BLOCK
    ELEMENTS_PER_BLOCK = BLOCK_SIZE // 4  # Assuming 4 elements per instruction

    # Load the input data into shared memory
    smem_x = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    row = tl.program_id(0)
    cols = tl.arange(0, ELEMENTS_PER_BLOCK)
    mask = cols < n_features
    x = tl.load(x_ptr + row * n_features + cols * 4, mask=mask, other=0.0).to(tl.float32)
    smem_x[cols * 4] = x

    # Compute the mean
    mean = tl.sum(smem_x, axis=0) / n_features
    mean = tl.broadcast_to(mean, (BLOCK_SIZE,))
    smem_x -= mean

    # Compute the variance
    var = tl.sum(smem_x * smem_x, axis=0) / n_features
    rstd = tl.rsqrt(var + 1e-5)
    rstd = tl.broadcast_to(rstd, (BLOCK_SIZE,))

    # Load the weight and bias
    w = tl.load(w_ptr + cols * 4, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols * 4, mask=mask, other=0.0).to(tl.float32)

    # Compute the output
    out = smem_x * rstd * w + b
    tl.store(out_ptr + row * n_features + cols * 4, out.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, w, b):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_features = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_features)
    _layernorm[(n_rows,)](x, w, b, out, n_features, BLOCK=BLOCK_SIZE)
    return out
