import torch
import triton
import triton.language as tl

@triton.jit
def cross_entropy_kernel(logits_ptr, target_ptr, output_ptr, M, V, BLOCK_SIZE_M: tl.constexpr):
    # Define the program's execution grid
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    
    # Calculate the row index
    row_idx = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask = row_idx < M
    
    # Load logits and targets
    logits = tl.load(logits_ptr + row_idx[:, None] * V + tl.arange(0, V), mask=mask[:, None])
    targets = tl.load(target_ptr + row_idx, mask=mask)
    
    # Convert targets to one-hot encoding
    target_one_hot = tl.zeros((BLOCK_SIZE_M, V), dtype=tl.float32)
    target_one_hot[tl.arange(0, BLOCK_SIZE_M), targets] = 1.0
    
    # Compute log softmax
    max_logit = tl.max(logits, axis=1, keepdim=True)
    log_sum_exp = tl.sum(tl.exp(logits - max_logit), axis=1, keepdim=True)
    log_softmax = logits - max_logit - tl.log(log_sum_exp)
    
    # Compute cross-entropy loss
    loss = -tl.sum(log_softmax * target_one_hot, axis=1)
    
    # Store the result
    tl.store(output_ptr + row_idx, loss, mask=mask)

def kernel(logits, target):
    M, V = logits.shape
    BLOCK_SIZE_M = 128
    output = torch.empty(M, dtype=torch.float32, device=logits.device)
    
    # Launch the Triton kernel
    grid = lambda opt: (triton.cdiv(M, BLOCK_SIZE_M), V)
    cross_entropy_kernel[grid](logits, target, output, M, V, BLOCK_SIZE_M=BLOCK_SIZE_M)
    
    return output
