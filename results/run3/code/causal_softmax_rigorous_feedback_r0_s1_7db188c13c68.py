import torch
import triton
import triton.language as tl

@triton.jit
def _causal_softmax(scores_ptr, out_ptr, T, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    mask = (cols > rows) & (rows < T) & (cols < T)
    
    # Load scores
    scores = tl.load(scores_ptr + batch * T * T + rows * T + cols, mask=mask, other=float('-inf')).to(tl.float32)
    
    # Compute max
    max_score = tl.max(scores, axis=1)
    
    # Subtract max for numerical stability
    scores = scores - max_score[:, None]
    
    # Exponentiate
    exp_scores = tl.exp(scores)
    
    # Sum of exponentials
    sum_exp_scores = tl.sum(exp_scores, axis=1)
    
    # Softmax
    softmax = exp_scores / sum_exp_scores[:, None]
    
    # Store results
    tl.store(out_ptr + batch * T * T + rows * T + cols, softmax.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(scores):
    scores = scores.contiguous()
    out = torch.empty_like(scores)
    B, T, _ = scores.shape
    BLOCK = triton.next_power_of_2(T)
    _causal_softmax[(B,)](scores, out, T, BLOCK=BLOCK)
    return out
