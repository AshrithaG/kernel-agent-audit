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

    # Mask out positions above the diagonal
    mask = col_range > row_range
    scores = tl.where(mask, -float('inf'), scores)

    # Compute max and exp
    row_max = tl.max(scores, axis=1)
    row_max = tl.broadcast_to(row_max[:, None], (BLOCK_SIZE, BLOCK_SIZE))
    scores = scores - row_max

    exp_scores = tl.exp(scores)

    # Compute sum of exps
    row_sum = tl.sum(exp_scores, axis=1)
    row_sum = tl.broadcast_to(row_sum[:, None], (BLOCK_SIZE, BLOCK_SIZE))

    # Compute softmax
    softmax = exp_scores / row_sum

    # Store results back to global memory
    tl.store(output_ptr + row_start + col_range * stride_2, softmax, mask=row_range < stride_0)

def kernel(scores):
    batch_size, T, _ = scores.shape
    output = torch.empty_like(scores)

    grid = (
        triton.cdiv(T, 32),
        triton.cdiv(T, 32),
        batch_size
    )

    causal_softmax_kernel[grid](
        scores.contiguous(), output.contiguous(),
        batch_size, T, scores.stride(0), scores.stride(1), scores.stride(2),
        BLOCK_SIZE=32
    )

    return output

# Example usage
if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    scores = torch.randn(1, 64, 64, dtype=torch.float16, device='cuda')
    ref_output = torch.softmax(scores.float().masked_fill(torch.triu(torch.ones(64, 64, dtype=torch.bool, device='cuda')).triu(1), float('-inf')), dim=-1).to(scores.dtype)
    output = kernel(scores)
    print("Reference output:", ref_output)
    print("Custom output:", output)
    print("Max diff:", torch.max(torch.abs(ref_output - output)))
