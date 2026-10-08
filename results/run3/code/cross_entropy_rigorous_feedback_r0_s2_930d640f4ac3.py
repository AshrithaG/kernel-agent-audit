import torch
import triton
import triton.language as tl
from triton.language import extra

@triton.jit
def _cross_entropy(logits_ptr, target_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    
    # Load logits and target
    logits = tl.load(logits_ptr + row * n_cols + cols, mask=mask, other=-float('inf')).to(tl.float32)
    target = tl.load(target_ptr + row, mask=cols == 0, other=-1)
    
    # Compute max logits for numerical stability
    max_logit = tl.max(logits, axis=0)
    logits -= max_logit
    
    # Compute logsumexp
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0)
    logsumexp = max_logit + tl.log(sum_exp_logits)
    
    # Compute loss
    target_logit = logits[target]
    loss = -target_logit + logsumexp
    
    # Store the result
    tl.store(out_ptr + row, loss.to(out_ptr.dtype.element_ty), mask=cols == 0)

def kernel(logits, target):
    logits = logits.contiguous().half()
    target = target.contiguous().int()
    out = torch.empty((logits.shape[0],), dtype=torch.float32, device=logits.device)
    _cross_entropy[(logits.shape[0],)](logits, target, out, logits.shape[1], BLOCK=triton.next_power_of_2(logits.shape[1]))
    return out
