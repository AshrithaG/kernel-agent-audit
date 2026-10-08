import torch
import triton
import triton.language as tl

@triton.jit
def _causal_softmax(scores_ptr, out_ptr, batch_size, seq_len, BLOCK: tl.constexpr):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    row = tl.program_id(2)
    cols = tl.arange(0, BLOCK)
    
    # Load scores into shared memory
    scores = tl.full((BLOCK,), -float('inf'), dtype=tl.float32)
    mask = (cols <= row) & (cols < seq_len)
    scores[cols[mask]] = tl.load(scores_ptr + batch_idx * seq_len * seq_len + head_idx * seq_len * seq_len + row * seq_len + cols[mask], mask=mask, other=-float('inf')).to(tl.float32)
    
    # Compute max for numerical stability
    max_score = tl.max(scores, axis=0)
    
    # Subtract max score for numerical stability
    scores -= max_score
    
    # Compute exp
    exp_scores = tl.exp(scores)
    
    # Compute sum of exp scores
    sum_exp_scores = tl.sum(exp_scores, axis=0)
    
    # Compute softmax
    softmax = exp_scores / sum_exp_scores
    
    # Store results
    tl.store(out_ptr + batch_idx * seq_len * seq_len + head_idx * seq_len * seq_len + row * seq_len + cols, softmax.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(scores):
    scores = scores.contiguous()
    batch_size, seq_len, _ = scores.shape
    out = torch.empty_like(scores)
    
    # Launch kernel
    grid = (batch_size, 1, seq_len)
    BLOCK = triton.next_power_of_2(seq_len)
    _causal_softmax[grid](scores, out, batch_size, seq_len, BLOCK=BLOCK)
    
    return out
