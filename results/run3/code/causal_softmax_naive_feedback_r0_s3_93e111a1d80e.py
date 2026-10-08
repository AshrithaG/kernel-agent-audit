import torch
import triton
import triton.language as tl

@triton.jit
def _causal_softmax(scores_ptr, out_ptr, T, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    mask = (cols > rows)[None, :]
    
    # Load scores into shared memory
    scores = tl.load(scores_ptr + batch * T * T + rows[:, None] * T + cols, mask=mask, other=float('-inf')).to(tl.float32)
    
    # Compute max for numerical stability
    max_score = tl.max(scores, axis=1, keepdims=True)
    scores = scores - max_score
    
    # Compute exp
    exp_scores = tl.exp(scores)
    
    # Compute sum of exps
    sum_exp_scores = tl.sum(exp_scores, axis=1, keepdims=True)
    
    # Compute softmax
    softmax = exp_scores / sum_exp_scores
    
    # Store results
    tl.store(out_ptr + batch * T * T + rows[:, None] * T + cols, softmax, mask=mask)

def kernel(scores):
    scores = scores.contiguous()
    T = scores.shape[-1]
    out = torch.empty_like(scores)
    grid = (scores.shape[0],)
    BLOCK = triton.next_power_of_2(T)
    _causal_softmax[grid](scores, out, T, BLOCK=BLOCK)
    return out
