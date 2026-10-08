import torch
import triton
import triton.language as tl

@triton.jit
def _causal_softmax(scores_ptr, out_ptr, n_rows, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    scores = tl.load(scores_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Mask out upper triangular part
    diag = tl.arange(0, BLOCK)
    upper_mask = cols > diag
    scores = tl.where(upper_mask, -float('inf'), scores)
    
    # Find max value in the row
    max_val = tl.max(scores, axis=1)
    max_val = tl.broadcast_to(max_val[:, None], (BLOCK,))
    
    # Subtract max value for numerical stability
    scores = scores - max_val
    
    # Compute exp
    exp_scores = tl.exp(scores)
    
    # Compute sum of exp values
    sum_exp_scores = tl.sum(exp_scores, axis=1)
    sum_exp_scores = tl.broadcast_to(sum_exp_scores[:, None], (BLOCK,))
    
    # Compute softmax
    softmax_scores = exp_scores / sum_exp_scores
    
    # Store the results
    tl.store(out_ptr + row * n_cols + cols, softmax_scores.to(out_ptr.dtype.element_ty), mask=mask)

def kernel(scores):
    scores = scores.contiguous()
    out = torch.empty_like(scores)
    n_rows, n_cols = scores.shape
    _causal_softmax[(n_rows,)](scores, out, n_rows, n_cols, BLOCK=1024)
    return out
