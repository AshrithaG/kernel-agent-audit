import torch
import triton
import triton.language as tl

@triton.jit
def _cross_entropy(logits_ptr, target_ptr, out_ptr, vocab_size, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < vocab_size
    
    logits = tl.load(logits_ptr + row * vocab_size + cols, mask=mask, other=0.0).to(tl.float32)
    target = tl.load(target_ptr + row, other=-1).to(tl.int32)
    
    # Compute log_softmax
    max_logit = tl.max(logits, axis=0)
    logits = logits - max_logit
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0)
    log_sum_exp_logits = tl.log(sum_exp_logits)
    log_softmax = logits - log_sum_exp_logits
    
    # Compute cross-entropy loss
    loss = -log_softmax[target]
    
    tl.store(out_ptr + row, loss.to(out_ptr.dtype.element_ty), mask=True)

def kernel(logits, target):
    logits = logits.contiguous()
    target = target.contiguous()
    out = torch.empty((logits.shape[0],), dtype=torch.float32, device=logits.device)
    _cross_entropy[(logits.shape[0],)](logits, target, out, logits.shape[1], BLOCK=triton.next_power_of_2(logits.shape[1]))
    return out
