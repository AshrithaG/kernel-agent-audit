import torch
import triton
import triton.language as tl

# Define the Triton kernel for causal softmax
@triton.jit
def causal_softmax_kernel(
    scores_ptr, output_ptr, seq_len, BLOCK_SIZE: tl.constexpr
):
    # Each thread processes one element in the last dimension
    row_id = tl.program_id(0) // BLOCK_SIZE
    col_id = tl.program_id(0) % BLOCK_SIZE

    # Load the input scores into shared memory
    if col_id <= row_id:
        score = tl.load(scores_ptr + row_id * seq_len + col_id)
    else:
        score = float('-inf')

    # Compute the sum of exponentials for the current row
    exp_score = tl.exp(score)
    sum_exp = tl.sum(exp_score, axis=1)

    # Compute the softmax values
    softmax_value = exp_score / sum_exp

    # Store the result back to the output tensor
    if col_id <= row_id:
        tl.store(output_ptr + row_id * seq_len + col_id, softmax_value)

# Define the wrapper function to call the Triton kernel
def kernel(scores):
    batch_size, seq_len, _ = scores.shape
    output = torch.empty_like(scores, dtype=torch.float32)

    # Launch the Triton kernel
    grid = (batch_size * seq_len, )
    BLOCK_SIZE = 128
    for b in range(batch_size):
        causal_softmax_kernel[grid](scores[b].contiguous(), output[b].contiguous(), seq_len, BLOCK_SIZE=BLOCK_SIZE)

    # Convert the output back to the original dtype
    return output.to(scores.dtype)

# Example usage
if __name__ == "__main__":
    import torch

    # Create a random float16 tensor
    scores = torch.rand((4, 1024, 1024), dtype=torch.float16, device='cuda')

    # Apply causal softmax using the reference implementation
    reference_output = causal_softmax(scores)

    # Apply causal softmax using the Triton kernel
    triton_output = kernel(scores)

    # Check if the outputs are close
    print(torch.allclose(reference_output, triton_output))
