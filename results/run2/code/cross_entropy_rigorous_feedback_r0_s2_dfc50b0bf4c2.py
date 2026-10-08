import torch
import triton
import triton.language as tl

@triton.jit
def cross_entropy_kernel(
    logits_ptr, target_ptr, output_ptr,
    M: tl.constexpr, V: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    # Each thread processes one row
    pid = tl.program_id(axis=0)
    row_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = row_idx < M

    # Load logits and target for this row
    logits = tl.load(logits_ptr + row_idx[:, None] * V + tl.arange(0, V), mask=mask[:, None])
    target = tl.load(target_ptr + row_idx, mask=mask)

    # Compute log softmax
    max_logit = tl.max(logits, axis=1)
    logits = logits - max_logit[:, None]
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=1)
    log_sum_exp_logits = tl.log(sum_exp_logits)
    log_softmax = logits - log_sum_exp_logits[:, None]

    # Gather the log softmax value corresponding to the target
    target_log_softmax = tl.gather(log_softmax, 1, target[:, None])

    # Compute negative log softmax
    neg_log_softmax = -target_log_softmax

    # Store the result
    tl.store(output_ptr + row_idx, neg_log_softmax, mask=mask)

def kernel(logits, target):
    M, V = logits.shape
    output = torch.empty(M, dtype=torch.float32, device=logits.device)

    BLOCK_M = 128  # Experiment with different block sizes for optimal performance

    grid = lambda META: (triton.cdiv(M, BLOCK_M),)
    cross_entropy_kernel[grid](logits, target, output, M, V, BLOCK_M)

    return output

# Example usage:
# logits = torch.randn(M, V, dtype=torch.float16, device='cuda')
# target = torch.randint(0, V, (M,), dtype=torch.int64, device='cuda')
# output = kernel(logits, target)
# print(output)
