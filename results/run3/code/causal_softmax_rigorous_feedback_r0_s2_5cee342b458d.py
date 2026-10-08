import torch
import triton
import triton.language as tl

@triton.jit
def _causal_softmax(scores_ptr, out_ptr, T, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    mask = (rows <= cols) & (cols < T)
    
    # Load scores
    scores = tl.load(scores_ptr + batch * T * T + rows * T + cols, mask=mask, other=float("-inf")).to(tl.float32)
    
    # Apply mask
    scores = tl.where(rows > cols, float("-inf"), scores)
    
    # Compute max
    max_val = tl.max(scores, axis=1)
    max_val = tl.broadcast_to(max_val, (BLOCK, BLOCK))
    
    # Subtract max for numerical stability
    scores = scores - max_val
    
    # Compute exp
    exp_scores = tl.exp(scores)
    
    # Compute sum of exps
    sum_exps = tl.sum(exp_scores, axis=1)
    sum_exps = tl.broadcast_to(sum_exps, (BLOCK, BLOCK))
    
    # Compute softmax
    softmax = exp_scores / sum_exps
    
    # Store results
    tl.store(out_ptr + batch * T * T + rows * T + cols, softmax.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(scores):
    scores = scores.contiguous()
    out = torch.empty_like(scores)
    B, T, _ = scores.shape
    _causal_softmax[(B,)](scores, out, T, BLOCK=T)
    return out
