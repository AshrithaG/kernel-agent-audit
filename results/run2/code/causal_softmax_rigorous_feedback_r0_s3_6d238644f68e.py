import torch
import triton
import triton.language as tl
import math

@triton.jit
def causal_softmax_kernel(
    scores_ptr, output_ptr, batch_size, seq_len, stride, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0) // seq_len
    col_idx = tl.program_id(0) % seq_len
    pid = tl.program_id(1)

    # Load scores into shared memory
    scores = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(pid, seq_len, BLOCK_SIZE):
        if i >= col_idx:
            scores[i - col_idx] = tl.load(scores_ptr + row_idx * stride + col_idx * seq_len + i, mask=i < seq_len, other=-float('inf'))

    # Compute max and sum for softmax
    max_val = tl.max(scores, axis=0)
    scores = scores - max_val
    exp_scores = tl.exp(scores)
    sum_exp_scores = tl.sum(exp_scores, axis=0)

    # Normalize
    norm_scores = exp_scores / sum_exp_scores

    # Store results back to global memory
    for i in range(pid, seq_len, BLOCK_SIZE):
        if i >= col_idx:
            tl.store(output_ptr + row_idx * stride + col_idx * seq_len + i, norm_scores[i - col_idx], mask=i < seq_len)

def kernel(scores):
    batch_size, seq_len, _ = scores.shape
    output = torch.empty_like(scores)
    grid = (batch_size * seq_len, triton.cdiv(seq_len, 32))
    causal_softmax_kernel[grid](scores, output, batch_size, seq_len, scores.stride(0), BLOCK_SIZE=32)
    return output

# Example usage:
# scores = torch.randn(8, 512, 512, dtype=torch.float16, device='cuda')
# result = kernel(scores)
# print(result)
