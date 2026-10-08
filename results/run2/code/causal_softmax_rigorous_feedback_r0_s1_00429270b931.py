import torch
import triton
import triton.language as tl

@triton.jit
def causal_softmax_kernel(
    scores_ptr, 
    output_ptr, 
    batch_size, 
    T, 
    BLOCK_SIZE: tl.constexpr
):
    # Get the thread ID
    row_idx = tl.program_id(0) // T
    col_idx = tl.program_id(0) % T
    
    # Load scores into shared memory
    scores_block = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    if row_idx < batch_size and col_idx < T:
        scores_block[col_idx] = tl.load(scores_ptr + row_idx * T * T + col_idx * T + col_idx, mask=(col_idx < T))
    
    # Mask out positions above the diagonal
    if col_idx > row_idx:
        scores_block[col_idx] = float('-inf')
    
    # Compute the max value for numerical stability
    max_score = tl.max(scores_block, axis=0)
    
    # Subtract max value from scores
    scores_block -= max_score
    
    # Exponentiate the scores
    exp_scores = tl.exp(scores_block)
    
    # Compute the sum of exponentiated scores
    sum_exp_scores = tl.sum(exp_scores, axis=0)
    
    # Divide by the sum to get the softmax probabilities
    softmax_probs = exp_scores / sum_exp_scores
    
    # Store the results back to global memory
    if row_idx < batch_size and col_idx < T:
        tl.store(output_ptr + row_idx * T * T + col_idx * T + col_idx, softmax_probs[col_idx], mask=(col_idx < T))

def kernel(scores):
    batch_size, T, _ = scores.shape
    output = torch.empty_like(scores)
    
    # Define the grid and block sizes
    grid = (batch_size * T,)
    block = (T,)
    
    # Launch the kernel
    causal_softmax_kernel[grid, block](
        scores.contiguous(),
        output.contiguous(),
        batch_size,
        T,
        T
    )
    
    return output

# Example usage
if __name__ == "__main__":
    import torch
    scores = torch.rand((8, 64, 64), dtype=torch.float16, device='cuda')
    result = kernel(scores)
    print(result)
