import torch
import triton
import triton.language as tl

@triton.jit
def causal_softmax_kernel(
    scores_ptr, mask_ptr, output_ptr,
    B, T,
    BLOCK_SIZE: tl.constexpr
):
    # Load scores into shared memory
    row_id = tl.program_id(axis=0) // T
    col_id = tl.program_id(axis=1)
    if row_id >= B or col_id >= T:
        return
    scores_block = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        idx = row_id * T + col_id + i * T
        if idx < B * T:
            scores_block[i] = tl.load(scores_ptr + idx, mask=(idx < B * T), other=-float('inf'))

    # Apply mask
    mask_block = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for i in range(BLOCK_SIZE):
        idx = row_id * T + col_id + i * T
        if idx < B * T:
            mask_block[i] = tl.load(mask_ptr + idx, mask=(idx < B * T), other=0)
    scores_block = tl.where(mask_block == 1, -float('inf'), scores_block)

    # Compute max and exp
    row_max = tl.max(scores_block, axis=0)
    row_max = tl.broadcast_to(row_max, (BLOCK_SIZE,))
    scores_block = scores_block - row_max
    exp_scores = tl.exp(scores_block)

    # Compute sum of exps
    sum_exp = tl.sum(exp_scores, axis=0)
    sum_exp = tl.broadcast_to(sum_exp, (BLOCK_SIZE,))

    # Compute softmax
    softmax_block = exp_scores / sum_exp

    # Store results back to global memory
    for i in range(BLOCK_SIZE):
        idx = row_id * T + col_id + i * T
        if idx < B * T:
            tl.store(output_ptr + idx, softmax_block[i], mask=(idx < B * T))

def kernel(scores):
    B, T = scores.shape
    output = torch.empty_like(scores)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=scores.device), 1)

    # Launch Triton kernel
    grid = lambda META: (
        triton.cdiv(B * T, META['BLOCK_SIZE']),
        T,
    )
    causal_softmax_kernel[grid](
        scores.contiguous(),
        mask.contiguous(),
        output.contiguous(),
        B, T,
        BLOCK_SIZE=128,
    )

    return output

# Example usage
if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    scores = torch.randn(1, 128, 128, dtype=torch.float16, device='cuda')
    start = time.time()
    result = kernel(scores)
    end = time.time()
    print(f"Time taken: {end - start:.4f} seconds")
    print(result)
