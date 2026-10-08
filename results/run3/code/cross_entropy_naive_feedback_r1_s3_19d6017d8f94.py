import torch
import triton
import triton.language as tl

@triton.jit
def _cross_entropy(logits_ptr, target_ptr, out_ptr, vocab_size, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < vocab_size
    
    # Load logits and target
    logits = tl.load(logits_ptr + row * vocab_size + cols, mask=mask, other=0.0).to(tl.float32)
    target = tl.load(target_ptr + row, mask=mask, other=0).to(tl.int32)
    
    # Compute log softmax
    max_logit = tl.max(logits, axis=0, mask=mask)
    logits_minus_max = logits - max_logit
    exp_logits = tl.exp(logits_minus_max)
    sum_exp_logits = tl.sum(exp_logits, axis=0, mask=mask)
    log_sum_exp_logits = tl.log(sum_exp_logits)
    log_softmax = logits_minus_max - log_sum_exp_logits
    
    # Compute cross-entropy loss
    target_logit = log_softmax[tl.where(cols == target, mask)]
    loss = -target_logit
    
    # Store the result
    tl.store(out_ptr + row, loss, mask=mask)

def kernel(logits, target):
    logits = logits.contiguous()
    target = target.contiguous()
    out = torch.empty(logits.shape[0], dtype=torch.float32, device=logits.device)
    vocab_size = logits.shape[1]
    _cross_entropy[(logits.shape[0],)](logits, target, out, vocab_size, BLOCK=triton.next_power_of_2(vocab_size))
    return out
