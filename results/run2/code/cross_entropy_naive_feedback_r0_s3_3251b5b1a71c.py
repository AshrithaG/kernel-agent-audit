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
    log_softmax = logits - max_logits[:, None] - tl.log(sum_exp_logits[:, None])
    
    # Gather target logits
    target_logit = log_softmax[tl.arange(0, BLOCK_M), targets]
    
    # Compute negative log likelihood
    neg_log_likelihood = -target_logit
    neg_log_likelihood = tl.where(mask, neg_log_likelihood, 0.0)
    
    # Store result
    tl.store(output_ptr + row_idx, neg_log_likelihood, mask=mask)

def kernel(logits, target):
    M, V = logits.shape
    output = torch.empty(M, dtype=torch.float32, device=logits.device)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        V,
    )
    
    cross_entropy_kernel[grid](
        logits.contiguous().half().data_ptr(),
        target.contiguous().data_ptr(),
        output.data_ptr(),
        M=M, V=V,
        BLOCK_M=128,
        num_warps=4,
        device=logits.device.index
    )
    
    return output

# Example usage
if __name__ == "__main__":
    import torch.nn.functional as F
    import time

    # Create random data
    M, V = 1024, 512
    logits = torch.randn(M, V, device='cuda', dtype=torch.float16)
    targets = torch.randint(0, V, (M,), device='cuda', dtype=torch.int64)

    # Reference implementation
    start_time = time.time()
    ref_output = F.cross_entropy(logits.float(), targets, reduction="none")
    ref_time = time.time() - start_time
    print(f"Reference time: {ref_time:.4f} seconds")

    # Triton implementation
    start_time = time.time()
    triton_output = kernel(logits, targets)
    triton_time = time.time() - start_time
    print(f"Triton time: {triton_time:.4f} seconds")

    # Check correctness
    assert torch.allclose(ref_output, triton_output, atol=1e-4)
    print("Output matches reference implementation.")
