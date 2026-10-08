import torch
import triton
import triton.language as tl

@triton.jit
def causal_softmax_kernel(
    scores_ptr, output_ptr, seq_len, BLOCK_SIZE: tl.constexpr
):
    # Load scores into shared memory
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    # Compute the row and column indices
    row = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = col_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask out upper triangle
    mask = col > row

    # Load scores into registers
    scores = tl.load(scores_ptr + row[:, None] * seq_len + col, mask=mask, other=float('-inf'))

    # Compute max and subtract from scores to avoid overflow
    row_max = tl.max(scores, axis=1)
    scores = scores - row_max[:, None]

    # Compute exp and sum
    exp_scores = tl.exp(scores)
    sum_exp_scores = tl.sum(exp_scores, axis=1)

    # Compute softmax
    softmax = exp_scores / sum_exp_scores[:, None]

    # Store results back to global memory
    tl.store(output_ptr + row[:, None] * seq_len + col, softmax, mask=mask)

def kernel(scores):
    batch_size, seq_len, _ = scores.shape
    output = torch.empty_like(scores)

    # Define grid and block sizes
    BLOCK_SIZE = 32
    num_rows = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_cols = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch kernel
    causal_softmax_kernel[triton.gemm_config(num_rows, num_cols, BLOCK_SIZE)](
        scores.contiguous(), output.contiguous(), seq_len, BLOCK_SIZE=BLOCK_SIZE
    )

    return output
