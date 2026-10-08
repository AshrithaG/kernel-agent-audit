import torch
import triton
import triton.language as tl
import math

@triton.jit
def _cross_entropy_kernel(
    logits_ptr, target_ptr, output_ptr, M, V,
    BLOCK_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    # Calculate row index
    row_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_idx < M

    # Load logits for the current row
    logits = tl.load(logits_ptr + row_idx[:, None] * V + tl.arange(0, V)[None, :], mask=row_mask[:, None], other=-float('inf'))

    # Find the max logit value for numerical stability
    max_logit = tl.max(logits, axis=1)
    max_logit = tl.where(row_mask, max_logit, -float('inf'))

    # Subtract max_logit from logits
    logits = logits - max_logit[:, None]

    # Compute the sum of exponentials
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=1)

    # Load target indices
    target = tl.load(target_ptr + row_idx, mask=row_mask, other=0)

    # Compute softmax values
    softmax_values = exp_logits / sum_exp_logits[:, None]
    softmax_values = tl.where(row_mask[:, None], softmax_values, 0.0)

    # Compute negative log likelihood
    neg_log_likelihood = -tl.log(softmax_values[tl.arange(0, BLOCK_M), target])
    neg_log_likelihood = tl.where(row_mask, neg_log_likelihood, 0.0)

    # Store the result
    tl.store(output_ptr + row_idx, neg_log_likelihood, mask=row_mask)

def kernel(logits, target):
    M, V = logits.shape
    assert target.shape == (M,)
    assert logits.dtype == torch.float16
    assert target.dtype == torch.int64

    # Allocate output tensor
    output = torch.empty(M, dtype=torch.float32, device=logits.device)

    # Define grid and block sizes
    BLOCK_M = 128
    grid = (triton.cdiv(M, BLOCK_M), 1)

    # Launch the kernel
    _cross_entropy_kernel[grid](
        logits.contiguous().data_ptr(),
        target.contiguous().data_ptr(),
        output.contiguous().data_ptr(),
        M, V,
        BLOCK_M=BLOCK_M
    )

    return output
