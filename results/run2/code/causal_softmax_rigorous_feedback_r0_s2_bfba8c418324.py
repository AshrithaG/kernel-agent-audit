import torch
import triton
import triton.language as tl
import math

@triton.jit
def causal_softmax_kernel(
    output_ptr, input_ptr, batch_size, seq_len, stride_batch, stride_seq, stride_head, BLOCK_SIZE: tl.constexpr
):
    # Get the thread index
    row = tl.program_id(0) // BLOCK_SIZE
    col = tl.program_id(0) % BLOCK_SIZE

    # Load the input data
    input_offset = row * stride_seq + col
    input_data = tl.load(input_ptr + input_offset, mask=row < seq_len & col < seq_len, other=-float('inf'))

    # Mask out the upper triangle
    if col > row:
        input_data = -float('inf')

    # Compute the max value for numerical stability
    max_value = tl.max(input_data, axis=0)

    # Subtract the max value to avoid overflow
    input_data = input_data - max_value

    # Exponentiate the input data
    exp_data = tl.exp(input_data)

    # Compute the sum of the exponentiated data
    sum_exp_data = tl.sum(exp_data, axis=0)

    # Compute the softmax values
    softmax_data = exp_data / sum_exp_data

    # Store the results
    tl.store(output_ptr + input_offset, softmax_data, mask=row < seq_len & col < seq_len)

def kernel(input):
    batch_size, seq_len, _ = input.shape
    output = torch.empty_like(input)
    
    grid = lambda META: (
        triton.cdiv(seq_len * seq_len, META['BLOCK_SIZE']),
    )
    
    causal_softmax_kernel[grid](
        output,
        input,
        batch_size,
        seq_len,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        BLOCK_SIZE=128,
    )
    
    return output

# Example usage:
# scores = torch.randn(1, 128, 128, dtype=torch.float16, device='cuda')
# result = kernel(scores)
# print(result)
