import torch
import triton
import triton.language as tl

@triton.jit
def _cross_entropy(logits_ptr, target_ptr, out_ptr, n_vocab, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_vocab
    
    # Load logits and targets
    logits = tl.load(logits_ptr + row * n_vocab + cols, mask=mask, other=0.0).to(tl.float32)
    target = tl.load(target_ptr + row, other=0.0)
    
    # Compute log softmax
    max_logit = tl.max(logits, axis=0)
    logits -= max_logit
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0)
    log_sum_exp_logits = tl.log(sum_exp_logits)
    
    # Compute cross-entropy loss
    target_logit = logits[tl.where(cols == target)[0][0]]
    loss = -target_logit + log_sum_exp_logits
    
    # Store the result
    tl.store(out_ptr + row, loss.to(out_ptr.dtype.element_ty))

def kernel(logits, target):
    logits = logits.contiguous().half()  # Ensure logits are in float16
    target = target.contiguous().int()   # Ensure targets are in int32
    out = torch.empty(logits.size(0), dtype=torch.float32, device=logits.device)
    n_rows, n_vocab = logits.shape
    _cross_entropy[(n_rows,)](logits, target, out, n_vocab, BLOCK=triton.next_power_of_2(n_vocab))
    return out
