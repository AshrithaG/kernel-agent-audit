import torch
import triton
import triton.language as tl

# Define the Triton kernel for causal softmax
@triton.jit
def causal_softmax_kernel(
    scores_ptr, mask_ptr, output_ptr, T,
    BLOCK_SIZE: tl.constexpr
):
    # Get the row and column indices
    row_id = tl.program_id(0) // BLOCK_SIZE
    col_id = tl.program_id(1)
    
    # Calculate the index
    idx = row_id * T + col_id
    
    # Load scores
    scores = tl.load(scores_ptr + idx, mask=col_id < T, other=-float('inf'))
    
    # Mask out the upper triangular part
    mask = tl.load(mask_ptr + idx, mask=col_id < T, other=True)
    scores = tl.where(mask, -float('inf'), scores)
    
    # Compute the max value for numerical stability
    max_score = tl.max(scores, axis=1)
    max_score = tl.broadcast_to(max_score, (BLOCK_SIZE, T))
    
    # Subtract the max value from scores
    scores = scores - max_score
    
    # Compute the exponential of the scores
    exp_scores = tl.exp(scores)
    
    # Compute the sum of exponentials
    sum_exp_scores = tl.sum(exp_scores, axis=1)
    sum_exp_scores = tl.broadcast_to(sum_exp_scores, (BLOCK_SIZE, T))
    
    # Compute the softmax
    softmax = exp_scores / sum_exp_scores
    
    # Store the result
    tl.store(output_ptr + idx, softmax, mask=col_id < T)

# Define the wrapper function to call the Triton kernel
def kernel(scores):
    B, T, _ = scores.shape
    output = torch.empty_like(scores)
    mask = torch.ones(T, T, dtype=torch.bool, device=scores.device).triu(1)
    
    # Launch the Triton kernel
    grid = lambda opt: (
        triton.cdiv(B * T, opt.BLOCK_SIZE),
        T,
    )
    causal_softmax_kernel[grid](
        scores.contiguous(), mask.contiguous(), output.contiguous(), T,
        BLOCK_SIZE=128,
    )
    
    return output

# Example usage
if __name__ == "__main__":
    import torch
    scores = torch.rand((1, 1024, 1024), dtype=torch.float16, device='cuda')
    result = kernel(scores)
    print(result)
