import torch
import triton
import triton.language as tl

# Define the Triton kernel for causal softmax
@triton.jit
def causal_softmax_kernel(
    scores_ptr, mask_ptr, output_ptr, T,
    BLOCK_SIZE: tl.constexpr
):
    # Each thread processes a row of the matrix
    row_id = tl.program_id(0)
    
    # Load the scores into shared memory
    scores = tl.zeros([BLOCK_SIZE, T], dtype=tl.float32)
    mask = tl.zeros([BLOCK_SIZE, T], dtype=tl.int32)
    
    # Load the scores and mask
    for col_id in range(0, T, BLOCK_SIZE):
        cols = tl.arange(0, BLOCK_SIZE)
        idx = row_id * T + col_id + cols
        mask[row_id, cols] = tl.load(mask_ptr + idx, mask=col_id + cols < T, other=0)
        scores[row_id, cols] = tl.load(scores_ptr + idx, mask=col_id + cols < T, other=-float('inf'))
    
    # Mask out the upper triangular part
    mask = mask & (tl.arange(0, T) <= tl.arange(0, T).unsqueeze(0))
    
    # Apply the mask
    scores = tl.where(mask, scores, -float('inf'))
    
    # Compute the max value for numerical stability
    max_score = tl.max(scores, axis=1)
    max_score = max_score[:, None]
    
    # Subtract the max value from scores
    scores = scores - max_score
    
    # Exponentiate the scores
    exp_scores = tl.exp(scores)
    
    # Sum the exponentiated scores
    sum_exp_scores = tl.sum(exp_scores, axis=1)
    sum_exp_scores = sum_exp_scores[:, None]
    
    # Compute the softmax
    softmax = exp_scores / sum_exp_scores
    
    # Store the result
    for col_id in range(0, T, BLOCK_SIZE):
        cols = tl.arange(0, BLOCK_SIZE)
        idx = row_id * T + col_id + cols
        tl.store(output_ptr + idx, softmax[row_id, cols], mask=col_id + cols < T)

def kernel(scores):
    # Get the shape of the input tensor
    B, T, _ = scores.shape
    
    # Create a mask for the upper triangular part
    mask = torch.ones(T, T, dtype=torch.int32, device=scores.device).triu(1)
    
    # Allocate output tensor
    output = torch.empty_like(scores)
    
    # Launch the Triton kernel
    grid = (B * T,)
    causal_softmax_kernel[grid](
        scores.contiguous(), mask.contiguous(), output.contiguous(),
        T=T,
        BLOCK_SIZE=128
    )
    
    return output

# Example usage
if __name__ == "__main__":
    # Create a random input tensor
    scores = torch.rand((2, 8, 8), dtype=torch.float16, device='cuda')
    
    # Apply the causal softmax
    result = kernel(scores)
    
    # Print the result
    print(result)
