import torch
import triton
import triton.language as tl
import math

@triton.jit
def causal_softmax_kernel(
    scores_ptr, output_ptr, batch_size, T, stride_0, stride_1, stride_2,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    # Load scores into shared memory
    row_start = row_idx * stride_0
    col_start = col_idx * stride_1
    row_end = row_start + stride_0
    col_end = col_start + stride_1

    row_range = tl.arange(0, stride_0)
    col_range = tl.arange(0, stride_1)

    scores = tl.load(scores_ptr + row_start + col_range * stride_2, mask=row_range < stride_0, other=-float('inf'))

    # Apply causal mask
    mask = col_range > row_range
    scores = tl.where(mask, scores, -float('inf'))

    # Compute max and exp
    max_score = tl.max(scores, axis=0)
    exp_scores = tl.exp(scores - max_score)

    # Compute sum of exps
    sum_exp_scores = tl.sum(exp_scores, axis=0)

    # Compute softmax
    softmax = exp_scores / sum_exp_scores

    # Store results
    tl.store(output_ptr + row_start + col_range * stride_2, softmax, mask=row_range < stride_0)

# Triton wrapper function
def causal_softmax_triton(scores):
    batch_size, T, _ = scores.shape
    output = torch.empty_like(scores)
    
    # Define grid and block dimensions
    grid = (T, T)
    block = (128, 1)

    # Launch kernel
    causal_softmax_kernel[grid](
        scores.data_ptr(), output.data_ptr(),
        batch_size, T, scores.stride(0), scores.stride(1), scores.stride(2),
        BLOCK_SIZE=block[0]
    )

    return output

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    scores = torch.randn(1, 128, 128, dtype=torch.float16, device='cuda')
    reference_output = causal_softmax(scores)
    triton_output = causal_softmax_triton(scores)

    print("Reference output:", reference_output)
    print("Triton output:", triton_output)
    print("Max diff:", torch.max(torch.abs(reference_output - triton_output)))
