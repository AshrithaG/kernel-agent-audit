import torch
import triton
import triton.language as tl

@triton.jit
def _cross_entropy(logits_ptr, target_ptr, out_ptr, vocab_size, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < vocab_size
    
    logits = tl.load(logits_ptr + row * vocab_size + cols, mask=mask, other=-float('inf')).to(tl.float32)
    target = tl.load(target_ptr + row, mask=cols[0], other=0).to(tl.int32)
    
    # Subtract the max logit to improve numerical stability
    max_logit = tl.max(logits, axis=0, keepdim=True)
    logits -= max_logit
    
    # Compute exp(logit) for each element
    exp_logits = tl.exp(logits)
    
    # Compute sum(exp(logit)) for the current row
    sum_exp_logits = tl.sum(exp_logits, axis=0, keepdim=True)
    
    # Compute log(sum(exp(logit)))
    log_sum_exp_logits = tl.log(sum_exp_logits)
    
    # Compute the logit of the target class
    target_logit = logits[target]
    
    # Compute the cross-entropy loss for the current row
    loss = -target_logit + log_sum_exp_logits
    
    # Store the result
    tl.store(out_ptr + row, loss.to(out_ptr.dtype.element_ty), mask=cols[0])

def kernel(logits, target):
    logits = logits.contiguous().half()
    target = target.contiguous().int()
    out = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.float32)
    batch_size, vocab_size = logits.shape
    _cross_entropy[(batch_size,)](logits, target, out, vocab_size, BLOCK=triton.next_power_of_2(vocab_size))
    return out
