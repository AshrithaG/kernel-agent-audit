import torch
import triton
import triton.language as tl

# Define the Triton kernel for causal softmax
@triton.jit
def causal_softmax_kernel(
    scores_ptr, output_ptr, seq_len, BLOCK_SIZE: tl.constexpr
):
    # Determine the row and column indices
    row_id = tl.program_id(0) // BLOCK_SIZE
    col_id = tl.program_id(0) % BLOCK_SIZE

    # Compute the position in the input tensor
    pos = row_id * seq_len + col_id

    # Load the score if within bounds, otherwise set to -inf
    score = tl.where(col_id > row_id, tl.load(scores_ptr + pos, mask=(col_id <= row_id), other=float('-inf')), float('-inf'))

    # Compute the sum of exponentials for the current row
    exp_score = tl.exp(score)
    sum_exp = tl.sum(exp_score, axis=1)

    # Compute the softmax values
    softmax_value = exp_score / sum_exp

    # Store the result back to the output tensor
    tl.store(output_ptr + pos, softmax_value, mask=(col_id <= row_id))

# Define the wrapper function to call the Triton kernel
def kernel(scores):
    seq_len = scores.shape[-1]
    output = torch.empty_like(scores, dtype=torch.float32)

    # Launch the Triton kernel
    grid = (seq_len * seq_len,)
    BLOCK_SIZE = 128
    causal_softmax_kernel[grid](scores.contiguous(), output.contiguous(), seq_len, BLOCK_SIZE=BLOCK_SIZE)

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
