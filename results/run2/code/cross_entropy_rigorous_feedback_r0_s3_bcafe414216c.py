import torch
import triton
import triton.language as tl

@triton.jit
def cross_entropy_kernel(
    logits_ptr, target_ptr, output_ptr,
    M: tl.constexpr, V: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    
    # Calculate row index
    row_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = row_idx < M
    
    # Load logits and targets
    logits = tl.load(logits_ptr + row_idx[:, None] * V + tl.arange(0, V)[None, :], mask=mask[:, None])
    targets = tl.load(target_ptr + row_idx, mask=mask)
    
    # Convert logits to float32
    logits = tl.cast(logits, tl.float32)
    
    # Compute log softmax
    max_logits = tl.max(logits, axis=1)
    exp_logits = tl.exp(logits - max_logits[:, None])
    sum_exp_logits = tl.sum(exp_logits, axis=1)
    log_sum_exp_logits = tl.log(sum_exp_logits)
    
    # Compute cross-entropy loss
    target_indices = targets[:, None] == tl.arange(0, V)[None, :]
    log_softmax = exp_logits / sum_exp_logits[:, None]
    log_softmax = tl.where(target_indices, log_softmax, tl.full(log_softmax.shape, -float('inf')))
    loss = -tl.sum(log_softmax, axis=1)
    
    # Store the result
    tl.store(output_ptr + row_idx, loss, mask=mask)

def kernel(logits, target):
    M, V = logits.shape
    BLOCK_M = 128
    
    # Allocate output tensor
    output = torch.empty(M, dtype=torch.float32, device=logits.device)
    
    # Launch kernel
    grid = (triton.cdiv(M, BLOCK_M), V)
    cross_entropy_kernel[grid](
        logits.data_ptr(),
        target.data_ptr(),
        output.data_ptr(),
        M=M, V=V,
        BLOCK_M=BLOCK_M,
        num_warps=4,
        in0_dtype=logits.dtype,
        in1_dtype=target.dtype,
        out_dtype=output.dtype
    )
    
    return output
