import torch
import triton
import triton.language as tl

@triton.jit
def _cross_entropy(logits_ptr, target_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    
    # Load logits and targets
    logits = tl.load(logits_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    target = tl.load(target_ptr + row, mask=cols[0], other=-1).to(tl.int32)
    
    # Compute log softmax
    max_logit = tl.max(logits, axis=0)
    logits -= max_logit
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0)
    log_softmax = logits - tl.log(sum_exp_logits)
    
    # Compute cross entropy
    target_logit = log_softmax[target]
    ce = -target_logit
    
    # Store the result
    tl.store(out_ptr + row, ce.to(out_ptr.dtype.element_ty), mask=cols[0])

def kernel(logits, target):
    logits = logits.contiguous()
    target = target.contiguous()
    out = torch.empty(logits.shape[0], dtype=torch.float32, device=logits.device)
    n_rows, n_cols = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    _cross_entropy[(n_rows,)](logits, target, out, n_cols, BLOCK=BLOCK_SIZE)
    return out
