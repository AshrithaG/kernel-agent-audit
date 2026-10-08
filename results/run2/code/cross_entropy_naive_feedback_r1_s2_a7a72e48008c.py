import torch
import triton
import triton.language as tl
import math

# Triton kernel for per-row cross-entropy loss
@triton.jit
def cross_entropy_kernel(logits_ptr, target_ptr, output_ptr, M, V, BLOCK_SIZE_M: tl.constexpr):
    # Each thread processes one row
    row_idx = tl.program_id(axis=0)
    if row_idx >= M:
        return
    
    # Load logits and target for this row
    logits = tl.load(logits_ptr + row_idx * V, mask=row_idx < M, other=-float('inf'))
    target = tl.load(target_ptr + row_idx, mask=row_idx < M, other=0)
    
    # Convert target to int32
    target = tl.where(row_idx < M, target, 0).to(tl.int32)
    
    # Compute log softmax
    max_logit = tl.max(logits, axis=0)
    logits -= max_logit
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0)
    log_softmax = logits - tl.log(sum_exp_logits)
    
    # Compute cross-entropy loss for this row
    loss = -log_softmax[target]
    
    # Store the result
    tl.store(output_ptr + row_idx, loss, mask=row_idx < M)

def kernel(logits, target):
    M, V = logits.shape
    output = torch.empty(M, dtype=torch.float32, device=logits.device)
    
    # Define grid and block sizes
    BLOCK_SIZE_M = 128
    grid = lambda META: (triton.cdiv(M, BLOCK_SIZE_M),)
    
    # Launch the kernel
    cross_entropy_kernel[grid](logits, target, output, M, V, BLOCK_SIZE_M)
    
    return output

# Example usage
if __name__ == "__main__":
    import torch.nn.functional as F
    torch.manual_seed(0)
    logits = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
    target = torch.randint(0, 1024, (1024,), dtype=torch.int64, device='cuda')
    
    reference_output = F.cross_entropy(logits.float(), target, reduction="none")
    custom_output = kernel(logits, target)
    
    print("Reference output:", reference_output)
    print("Custom output:", custom_output)
    print("Are outputs close?", torch.allclose(reference_output, custom_output, atol=1e-5))
