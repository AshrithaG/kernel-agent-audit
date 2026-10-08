import torch
import triton
import triton.language as tl

@triton.jit
def _cross_entropy(logits_ptr, target_ptr, out_ptr, vocab_size, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < vocab_size
    
    # Load logits and target
    logits = tl.load(logits_ptr + row * vocab_size + cols, mask=mask, other=-float('inf')).to(tl.float32)
    target = tl.load(target_ptr + row, other=-1)
    
    # Compute log softmax
    max_logit = tl.max(logits, axis=0)
    logits -= max_logit
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0)
    log_sum_exp_logits = tl.log(sum_exp_logits)
    
    # Compute cross entropy loss
    target_logit = logits[tl.where(cols == target)]
    loss = -target_logit + log_sum_exp_logits
    
    # Store the result
    tl.store(out_ptr + row, loss, mask=True)

def kernel(logits, target):
    logits = logits.contiguous()
    target = target.contiguous()
    out = torch.empty(logits.size(0), dtype=torch.float32, device=logits.device)
    batch_size, vocab_size = logits.shape
    _cross_entropy[(batch_size,)](logits, target, out, vocab_size, BLOCK=triton.next_power_of_2(vocab_size))
    return out
