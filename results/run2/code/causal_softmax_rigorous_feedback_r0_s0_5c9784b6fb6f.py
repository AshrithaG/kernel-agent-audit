import torch
import triton
import triton.language as tl

@triton.jit
def causal_softmax_kernel(
    scores_ptr, output_ptr, seq_len, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    # Load scores into shared memory
    row_start = row_idx * BLOCK_SIZE
    col_start = col_idx * BLOCK_SIZE
    scores = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for i in range(0, BLOCK_SIZE, 32):
        if row_start + i < seq_len and col_start + i < seq_len:
            scores[i:i+32, i:i+32] = tl.load(
                scores_ptr + (row_start + i) * seq_len + (col_start + i),
                mask=(row_start + i < seq_len) & (col_start + i < seq_len),
                other=float('-inf')
            )

    # Mask out upper triangle
    for i in range(BLOCK_SIZE):
        for j in range(i + 1, BLOCK_SIZE):
            scores[i, j] = float('-inf')

    # Compute max and exp
    row_max = tl.max(scores, axis=1)
    row_max = tl.where(row_max == float('-inf'), 0.0, row_max)
    exp_scores = tl.exp(scores - row_max[:, None])

    # Compute sum of exps
    row_sum = tl.sum(exp_scores, axis=1)

    # Compute softmax
    softmax = exp_scores / row_sum[:, None]

    # Store results back to global memory
    for i in range(0, BLOCK_SIZE, 32):
        if row_start + i < seq_len and col_start + i < seq_len:
            tl.store(
                output_ptr + (row_start + i) * seq_len + (col_start + i),
                softmax[i:i+32, i:i+32],
                mask=(row_start + i < seq_len) & (col_start + i < seq_len)
            )

def kernel(scores):
    seq_len = scores.shape[1]
    output = torch.empty_like(scores)
    grid = (triton.cdiv(seq_len, 32), triton.cdiv(seq_len, 32))
    causal_softmax_kernel[grid](scores, output, seq_len, BLOCK_SIZE=32)
    return output

# Example usage
if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    scores = torch.rand((8, 1024, 1024), dtype=torch.float16, device='cuda')
    
    # Reference implementation
    start_time = time.time()
    ref_output = causal_softmax(scores)
    ref_time = time.time() - start_time
    print(f"Reference time: {ref_time:.4f} seconds")

    # Triton implementation
    start_time = time.time()
    triton_output = kernel(scores)
    triton_time = time.time() - start_time
    print(f"Triton time: {triton_time:.4f} seconds")

    # Check correctness
    assert torch.allclose(ref_output, triton_output, atol=1e-3, rtol=1e-3)
