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

    # Compute the position in the flattened tensor
    pos = row_idx * seq_len + col_idx

    # Load the scores into a register
    scores = tl.load(scores_ptr + pos, mask=col_idx < seq_len, other=float('-inf'))

    # Mask out the upper triangular part
    if col_idx > row_idx:
        scores = float('-inf')

    # Compute the max score in the row
    row_max = tl.max(scores, axis=1)
    row_max = tl.sum(row_max, axis=0)  # Broadcast to all threads in the block

    # Subtract the max score from the scores to avoid overflow
    scores -= row_max

    # Compute the exponential of the scores
    exp_scores = tl.exp(scores)

    # Compute the sum of the exponentials in the row
    row_sum = tl.sum(exp_scores, axis=1)
    row_sum = tl.sum(row_sum, axis=0)  # Broadcast to all threads in the block

    # Compute the softmax values
    softmax_values = exp_scores / row_sum

    # Store the softmax values back to the output tensor
    tl.store(output_ptr + pos, softmax_values, mask=col_idx < seq_len)

# Triton kernel wrapper
def kernel(scores):
    batch_size, seq_len, _ = scores.shape
    output = torch.empty_like(scores)
    
    grid = (
        triton.cdiv(seq_len, 128),
        triton.cdiv(seq_len, 128),
        batch_size
    )
    
    causal_softmax_kernel[grid](
        scores.contiguous(),
        output.contiguous(),
        seq_len,
        BLOCK_SIZE=128
    )
    
    return output

# Example usage
if __name__ == "__main__":
    import time

    # Create a random input tensor
    batch_size = 8
    seq_len = 512
    scores = torch.rand((batch_size, seq_len, seq_len), dtype=torch.float16, device='cuda')

    # Measure the execution time of the reference implementation
    start_time = time.time()
    ref_output = causal_softmax(scores)
    ref_time = time.time() - start_time
    print(f"Reference implementation time: {ref_time:.4f} seconds")

    # Measure the execution time of the Triton kernel
    start_time = time.time()
    triton_output = kernel(scores)
    triton_time = time.time() - start_time
    print(f"Triton kernel time: {triton_time:.4f} seconds")

    # Check if the outputs are the same within floating-point tolerance
    assert torch.allclose(ref_output, triton_output, atol=1e-3, rtol=1e-3)
    print("Outputs match within tolerance")
