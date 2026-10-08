import torch
import triton
import triton.language as tl

@triton.jit
def _causal_softmax(scores_ptr, out_ptr, batch_size, seq_len, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    col = tl.program_id(1)
    
    cols = tl.arange(0, BLOCK)
    mask = cols < seq_len
    
    scores = tl.load(scores_ptr + batch * seq_len * seq_len + col * seq_len + cols, mask=mask, other=float('-inf')).to(tl.float32)
    
    # Mask out upper triangle
    row = tl.arange(0, BLOCK)
    upper_triangle_mask = row > cols
    scores = tl.where(upper_triangle_mask, float('-inf'), scores)
    
    # Compute max for numerical stability
    max_score = tl.max(scores, axis=1)
    
    # Subtract max score for numerical stability
    scores -= max_score[:, None]
    
    # Compute exp
    exp_scores = tl.exp(scores)
    
    # Compute sum of exps
    sum_exp_scores = tl.sum(exp_scores, axis=1)
    
    # Compute softmax
    softmax_scores = exp_scores / sum_exp_scores[:, None]
    
    # Store the result
    tl.store(out_ptr + batch * seq_len * seq_len + col * seq_len + cols, softmax_scores.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(scores):
    scores = scores.contiguous()
    batch_size, seq_len, _ = scores.shape
    out = torch.empty_like(scores)
    BLOCK = triton.next_power_of_2(seq_len)
    _causal_softmax[(batch_size, seq_len // BLOCK)](scores, out, batch_size, seq_len, BLOCK=BLOCK)
    return out
