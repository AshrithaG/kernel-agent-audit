import torch
import triton
import triton.language as tl
import math

@triton.jit
def _cross_entropy_kernel(logits_ptr, target_ptr, output_ptr, M: tl.constexpr, V: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_id = tl.load(target_ptr + pid)
    row_offset = pid * V
    logits_row = tl.load(logits_ptr + row_offset, mask=row_id < V, other=-float('inf'))
    max_logit = tl.max(logits_row)
    logits_row = logits_row - max_logit
    exp_logits = tl.exp(logits_row)
    sum_exp = tl.sum(exp_logits)
    log_sum_exp = tl.log(sum_exp)
    target_logit = logits_row[row_id]
    loss = -target_logit + log_sum_exp
    tl.store(output_ptr + pid, loss)

def kernel(logits, target):
    M, V = logits.shape
    output = torch.empty(M, dtype=torch.float32, device=logits.device)
    grid = lambda META: (M,)
    BLOCK_SIZE = 1024
    _cross_entropy_kernel[grid](logits, target, output, M=M, V=V, BLOCK_SIZE=BLOCK_SIZE)
    return output
