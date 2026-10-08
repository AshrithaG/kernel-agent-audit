import torch
import triton
import triton.language as tl
import math

# Triton kernel for computing per-row cross-entropy loss
@triton.jit
def cross_entropy_kernel(
    logits_ptr, target_ptr, output_ptr, M, V,
    BLOCK_M: tl.constexpr
):
    # Define the thread indices
    row_idx = tl.program_id(0)
    col_idx = tl.load(target_ptr + row_idx)

    # Load the logit corresponding to the target class
    if row_idx < M:
        logit = tl.load(logits_ptr + row_idx * V + col_idx)
        # Compute the negative log of the softmax probability
        output = -tl.log(tl.exp(logit) / tl.sum(tl.exp(logits_ptr + row_idx * V), axis=0))
        # Store the result in the output tensor
        tl.store(output_ptr + row_idx, output)

# Function to compute cross-entropy loss using the Triton kernel
def kernel(logits, target):
    M, V = logits.shape
    output = torch.empty(M, dtype=torch.float32, device=logits.device)

    # Launch the Triton kernel
    grid = lambda opt: (triton.cdiv(M, opt.BLOCK_M),)
    cross_entropy_kernel[grid](logits, target, output, M, V, BLOCK_M=1024)

    return output

# Example usage
if __name__ == "__main__":
    import torch.nn.functional as F

    # Create random logits and target
    M, V = 1024, 512
    logits = torch.randn(M, V, device='cuda', dtype=torch.float16)
    target = torch.randint(0, V, (M,), device='cuda', dtype=torch.int64)

    # Compute cross-entropy loss using the Triton kernel
    triton_output = kernel(logits, target)

    # Compute cross-entropy loss using PyTorch's reference implementation
    reference_output = F.cross_entropy(logits.float(), target, reduction="none")

    # Check if the outputs are equal within a small tolerance
    assert torch.allclose(triton_output, reference_output, atol=1e-5), "Outputs do not match"

    print("Cross-entropy loss computed successfully using Triton kernel!")
