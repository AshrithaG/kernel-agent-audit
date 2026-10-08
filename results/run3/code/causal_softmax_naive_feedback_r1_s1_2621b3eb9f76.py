import torch
import triton
import triton.language as tl

@triton.jit
def _causal_softmax(scores_ptr, out_ptr, batch_size, seq_len, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    row = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    
    # Load scores
    scores_offset = batch * seq_len * seq_len + row * seq_len
    scores = tl.load(scores_ptr + scores_offset + cols, mask=cols < seq_len, other=float('-inf')).to(tl.float32)
    
    # Mask out upper triangle
    mask = tl.where(cols > row, float('-inf'), 0.0)
    scores = scores + mask
    
    # Compute max for numerical stability
    max_score = tl.max(scores, axis=0)
    
    # Subtract max score for numerical stability
    scores = scores - max_score
    
    # Exponentiate
    exp_scores = tl.exp(scores)
    
    # Sum of exponentials using a more efficient reduction
    sum_exp_scores = tl.sum(exp_scores, axis=0)
    
    # Softmax
    softmax = exp_scores / sum_exp_scores
    
    # Store results
    tl.store(out_ptr + scores_offset + cols, softmax.to(out_ptr.dtype.element_ty), mask=cols < seq_len)

def kernel(scores):
    scores = scores.contiguous()
    batch_size, seq_len, _ = scores.shape
    out = torch.empty_like(scores)
    grid = (batch_size, seq_len)
    BLOCK = triton.next_power_of_2(seq_len)
    _causal_softmax[grid](scores, out, batch_size, seq_len, BLOCK=BLOCK)
    return out
