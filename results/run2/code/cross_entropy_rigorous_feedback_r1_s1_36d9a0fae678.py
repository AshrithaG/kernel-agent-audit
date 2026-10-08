import torch
import triton
import triton.language as tl

@triton.jit
def cross_entropy_kernel(logits_ptr, target_ptr, output_ptr, M: tl.constexpr, V: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Calculate the row ID for this thread
    row_id = tl.program_id(axis=0) * BLOCK_SIZE + tl threadIdx.x
    # Mask to ensure we don't go out of bounds
    mask = row_id < M

    # Load logits for the current row
    logits = tl.load(logits_ptr + row_id * V, mask=mask, other=-float('inf'))

    # Find the max logit to improve numerical stability
    max_logit = tl.max(logits, axis=0)
    logits = logits - max_logit

    # Compute the sum of exponentials
    exp_logits = tl.exp(logits)
    sum_exp = tl.sum(exp_logits, axis=0)

    # Compute the log of the sum of exponentials
    log_sum_exp = tl.log(sum_exp)

    # Compute the negative log of the correct class probability
    target = tl.load(target_ptr + row_id, mask=mask, other=0)
    correct_class_logit = logits[target]
    loss = -correct_class_logit + log_sum_exp

    # Store the result
    tl.store(output_ptr + row_id, loss, mask=mask)

def kernel(logits, target):
    M, V = logits.shape
    assert logits.dtype == torch.float16, "Logits must be of type float16"
    assert target.dtype == torch.int64, "Targets must be of type int64"
    assert logits.is_cuda and target.is_cuda, "Inputs must be on CUDA"

    output = torch.empty(M, dtype=torch.float32, device=logits.device)

    # Define the grid
    BLOCK_SIZE = 1024
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE']),)

    # Launch the kernel
    cross_entropy_kernel[grid](logits.contiguous(), target.contiguous(), output.contiguous(), M=M, V=V, BLOCK_SIZE=BLOCK_SIZE)

    return output

# Example usage
if __name__ == "__main__":
    import torch.nn.functional as F

    # Generate random data
    M, V = 1024, 32000
    logits = torch.rand((M, V), dtype=torch.float16, device='cuda')
    target = torch.randint(0, V, (M,), dtype=torch.int64, device='cuda')

    # Reference implementation
    ref_output = F.cross_entropy(logits.float(), target, reduction="none")

    # Custom kernel implementation
    custom_output = kernel(logits, target)

    # Check if outputs are the same
    assert torch.allclose(ref_output, custom_output, atol=1e-5), "Outputs do not match"

    print("Reference output:", ref_output[:5])
    print("Custom output:", custom_output[:5])
