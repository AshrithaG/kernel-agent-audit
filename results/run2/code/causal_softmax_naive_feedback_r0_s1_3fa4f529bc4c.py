import torch
import triton
import triton.language as tl
import math

# Triton kernel for causal softmax
@triton.jit
def causal_softmax_kernel(
    scores_ptr, 
    output_ptr, 
    batch_size, 
    seq_len, 
    BLOCK_SIZE: tl.constexpr,
):
    # Define the program ID
    row = tl.program_id(0) // seq_len
    col = tl.program_id(0) % seq_len
    
    # Load scores into shared memory
    pid = tl.load(scores_ptr + row * seq_len * seq_len + col * seq_len, mask=col < seq_len, other=-float('inf'))
    
    # Apply causal mask
    if col > row:
        pid = -float('inf')
    
    # Compute max value
    max_val = tl.max(pid, axis=0)
    
    # Subtract max value for numerical stability
    pid -= max_val
    
    # Exponentiate
    exp_pid = tl.exp(pid)
    
    # Compute sum of exponentials
    sum_exp_pid = tl.sum(exp_pid, axis=0)
    
    # Compute softmax
    softmax_pid = exp_pid / sum_exp_pid
    
    # Store result
    tl.store(output_ptr + row * seq_len * seq_len + col * seq_len, softmax_pid, mask=col < seq_len)

# Wrapper function to call the Triton kernel
def kernel(scores):
    batch_size, seq_len, _ = scores.shape
    output = torch.empty_like(scores)
    
    # Launch the Triton kernel
    grid = lambda META: (batch_size * seq_len,)
    causal_softmax_kernel[grid](scores, output, batch_size, seq_len, BLOCK_SIZE=128)
    
    return output

# Example usage
if __name__ == "__main__":
    # Create a random tensor of shape [batch, T, T] in float16
    batch_size = 2
    seq_len = 32
    scores = torch.rand((batch_size, seq_len, seq_len), dtype=torch.float16, device='cuda')
    
    # Compute causal softmax using the Triton kernel
    result = kernel(scores)
    
    # Print the result
    print(result)
