import torch
import triton
import triton.language as tl

@triton.jit
def _causal_softmax(scores_ptr, out_ptr, seq_len, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    mask = (rows[:, None] <= cols[None, :]) & (cols < seq_len)
    
    # Load scores
    scores = tl.load(scores_ptr + batch * seq_len * seq_len + rows[:, None] * seq_len + cols[None, :], mask=mask, other=float('-inf')).to(tl.float32)
    
    # Compute max
    max_score = tl.max(scores, axis=1)
    max_score = tl.broadcast_to(max_score[:, None], scores.shape)
    
    # Subtract max for numerical stability
    scores = scores - max_score
    
    # Compute exp
    exp_scores = tl.exp(scores)
    
    # Compute sum of exp
    sum_exp_scores = tl.sum(exp_scores, axis=1)
    sum_exp_scores = tl.broadcast_to(sum_exp_scores[:, None], scores.shape)
    
    # Compute softmax
    softmax = exp_scores / sum_exp_scores
    
    # Store result
    tl.store(out_ptr + batch * seq_len * seq_len + rows[:, None] * seq_len + cols[None, :], softmax, mask=mask)

def kernel(scores):
    scores = scores.contiguous()
    batch_size, seq_len, _ = scores.shape
    out = torch.empty_like(scores)
    grid = (batch_size,)
    BLOCK = triton.next_power_of_2(seq_len)
    _causal_softmax[grid](scores, out, seq_len, BLOCK=BLOCK)
    return out
