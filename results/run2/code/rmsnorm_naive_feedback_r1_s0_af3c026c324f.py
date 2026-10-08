import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_elements, eps, BLOCK_SIZE: tl.constexpr):
    # Get the program ID and calculate the start index for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    indices = block_start + tl.arange(0, BLOCK_SIZE)
    mask = indices < n_elements

    # Load the relevant elements from x and w
    x = tl.load(x_ptr + indices, mask=mask, other=0.0)
    w = tl.load(w_ptr + indices % BLOCK_SIZE, mask=mask, other=0.0)

    # Compute the sum of squares for each row
    squared_sum = tl.sum(x * x, axis=1)
    # Compute the root mean square with epsilon to avoid division by zero
    rms = tl.sqrt(squared_sum / BLOCK_SIZE + eps)
    # Normalize x by rms and multiply by the weight vector
    normalized_x = x / rms
    result = normalized_x * w

    # Store the result back into the output tensor
    tl.store(out_ptr + indices, result, mask=mask)

def kernel(x, w, eps=1e-6):
    assert x.ndim == 2, "Input tensor must be 2-dimensional"
    assert w.ndim == 1, "Weight vector must be 1-dimensional"
    assert x.shape[1] == w.shape[0], "Last dimension of x must match length of w"

    n_elements = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(x.shape[1])

    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    rmsnorm_kernel[grid](x, w, out, n_elements, eps, BLOCK_SIZE)

    return out

# Example usage:
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(128, 512, device='cuda', dtype=torch.float16)
    w = torch.randn(512, device='cuda', dtype=torch.float16)

    result = kernel(x, w)
    print(result)
