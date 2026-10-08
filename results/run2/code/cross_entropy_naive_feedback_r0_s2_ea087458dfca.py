import torch
import triton
import triton.language as tl
import math

@triton.jit
def _cross_entropy_kernel(logits_ptr, target_ptr, output_ptr, M, V, BLOCK_M: tl.constexpr):
    # Get the row index and column index of this thread
    row_idx = tl.program_id(0)
    col_idx = tl.load(target_ptr + row_idx)

    # Load the logits for this row into shared memory
    row_offset = row_idx * V
    logits_row = tl.load(logits_ptr + row_offset, mask=col_idx < V, other=-float('inf'))

    # Compute the max logit value to improve numerical stability
    max_logit = tl.max(logits_row, axis=0)
    logits_row = logits_row - max_logit

    # Compute the sum of exponentials
    exp_sum = tl.sum(tl.exp(logits_row), axis=0)

    # Compute the log of the sum of exponentials
    log_exp_sum = tl.log(exp_sum)

    # Compute the cross-entropy loss for this row
    loss = -logits_row[col_idx] + log_exp_sum

    # Store the result in the output tensor
    if row_idx < M:
        tl.store(output_ptr + row_idx, loss.to(tl.float32))

def kernel(logits, target):
    M, V = logits.shape
    output = torch.empty(M, dtype=torch.float32, device=logits.device)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)
    _cross_entropy_kernel[grid](logits, target, output, M, V, BLOCK_M=1024)
    
    return output
