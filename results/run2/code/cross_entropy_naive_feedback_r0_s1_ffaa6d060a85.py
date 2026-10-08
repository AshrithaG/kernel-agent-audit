import torch
import triton
import triton.language as tl
import math

# Triton kernel for per-row cross-entropy loss
@triton.jit
def cross_entropy_kernel(logits_ptr, target_ptr, output_ptr, M, V, BLOCK_SIZE_M: tl.constexpr):
    # Define the row index and column index
    row_idx = tl.program_id(0)
    col_idx = tl.load(target_ptr + row_idx)

    # Load the logits for the current row
    logits = tl.load(logits_ptr + row_idx * V + tl.arange(0, V))

    # Get the logit value for the target class
    target_logit = logits[col_idx]

    # Compute the sum of logits for the current row
    sum_logits = tl.sum(tl.exp(logits - target_logit), axis=0)

    # Compute the loss for the current row
    loss = target_logit - tl.log(sum_logits + 1e-20)

    # Store the result in the output array
    tl.store(output_ptr + row_idx, loss)

# Wrapper function to call the Triton kernel
def kernel(logits, target):
    M, V = logits.shape
    output = torch.empty(M, dtype=torch.float32, device=logits.device)

    # Define the grid and block sizes
    grid = lambda opt: (triton.cdiv(M, opt.BLOCK_SIZE_M),)
    block = lambda opt: (opt.BLOCK_SIZE_M,)

    # Launch the Triton kernel
    cross_entropy_kernel[grid](logits.data_ptr(), target.data_ptr(), output.data_ptr(), M, V, BLOCK_SIZE_M=128)

    return output

# Example usage
if __name__ == "__main__":
    # Create random logits and target
    M, V = 1024, 1000
    logits = torch.randn(M, V, dtype=torch.float16, device='cuda')
    target = torch.randint(0, V, (M,), dtype=torch.int64, device='cuda')

    # Compute the cross-entropy loss using the Triton kernel
    loss = kernel(logits, target)

    # Print the first few losses
    print(loss[:5])
