import torch
import triton
import triton.language as tl
import math

# Triton kernel for per-row cross-entropy loss
@triton.jit
def cross_entropy_kernel(
    logits_ptr, target_ptr, output_ptr, M, V,
    BLOCK_M: tl.constexpr,
    BLOCK_V: tl.constexpr
):
    # Define the position of the current thread within the grid
    pid_m = tl.program_id(0)
    pid_v = tl.program_id(1)

    # Load the logits and target for this row and column
    logits = tl.load(logits_ptr + pid_m * V + pid_v, mask=pid_v < V, other=-float('inf'))
    target = tl.load(target_ptr + pid_m, mask=pid_m < M, other=-1)

    # Compute the log of the softmax probabilities
    max_logit = tl.max(logits, axis=1)
    logits -= max_logit
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=1)
    log_prob = logits[target] - tl.log(sum_exp_logits)

    # Store the result
    if pid_v == 0:
        tl.store(output_ptr + pid_m, -log_prob, mask=pid_m < M)

# Function to call the Triton kernel
def kernel(logits, target):
    M, V = logits.shape
    output = torch.empty(M, dtype=torch.float32, device=logits.device)

    # Define block sizes
    BLOCK_M = 128
    BLOCK_V = 128

    # Launch the Triton kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(V, BLOCK_V))
    cross_entropy_kernel[grid](
        logits, target, output,
        M, V,
        BLOCK_M=BLOCK_M,
        BLOCK_V=BLOCK_V,
    )

    return output

# Example usage
if __name__ == "__main__":
    import torch.nn.functional as F

    # Create example inputs
    logits = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
    target = torch.randint(0, 1024, (1024,), dtype=torch.int64, device='cuda')

    # Reference implementation
    reference_output = F.cross_entropy(logits.float(), target, reduction="none")

    # Triton implementation
    triton_output = kernel(logits, target)

    # Check if the outputs are the same
    assert torch.allclose(reference_output, triton_output, atol=1e-5, rtol=1e-5)

    print("Outputs match!")
