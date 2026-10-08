import torch
import triton
import triton.language as tl

@triton.jit
def causal_softmax_kernel(
    scores_ptr, output_ptr, batch_size, seq_len, BLOCK_SIZE: tl.constexpr
):
    # Get the program ID
    pid = tl.program_id(0)
    bid = tl.program_id(1)

    # Calculate the row and column IDs
    row_id = pid // BLOCK_SIZE
    col_id = pid % BLOCK_SIZE
    batch_id = bid

    # Calculate the position in the global memory
    offset = batch_id * seq_len * seq_len + row_id * seq_len + col_id

    # Load the score
    score = tl.load(scores_ptr + offset, mask=row_id < seq_len & col_id < seq_len, other=float("-inf"))

    # Mask out elements above the diagonal
    if col_id > row_id:
        score = float("-inf")

    # Store the masked score back to shared memory
    tl.store(output_ptr + offset, score)

    # Compute the max score for the row
    row_max = tl.max(score, axis=1)
    row_max = tl.sum(row_max, axis=0)

    # Subtract the max score from the current score
    score -= row_max

    # Compute the exponential of the score
    exp_score = tl.exp(score)

    # Compute the sum of exponentials for the row
    row_sum = tl.sum(exp_score, axis=1)
    row_sum = tl.sum(row_sum, axis=0)

    # Compute the softmax value
    softmax_value = exp_score / row_sum

    # Store the softmax value back to global memory
    tl.store(output_ptr + offset, softmax_value, mask=row_id < seq_len & col_id < seq_len)

def kernel(scores):
    batch_size, seq_len, _ = scores.shape
    output = torch.empty_like(scores)

    # Define the block size
    BLOCK_SIZE = 32

    # Launch the kernel
    grid = (seq_len * BLOCK_SIZE, batch_size)
    causal_softmax_kernel[grid](
        scores.contiguous(), output.contiguous(), batch_size, seq_len, BLOCK_SIZE=BLOCK_SIZE
    )

    return output

# Example usage
if __name__ == "__main__":
    import torch

    # Create a random float16 tensor of shape [batch, T, T]
    batch_size = 4
    seq_len = 1024
    scores = torch.rand((batch_size, seq_len, seq_len), dtype=torch.float16, device="cuda")

    # Apply the causal softmax
    result = kernel(scores)

    # Print the result
    print(result)
