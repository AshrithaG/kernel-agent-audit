import torch
import triton
import triton.language as tl

@triton.jit
def causal_softmax_kernel(
    scores_ptr, 
    output_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
    CACHE_LINE_SIZE: tl.constexpr
):
    # Define the position in the batch and sequence
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    
    # Calculate the row and column indices
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = bid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask out elements above the diagonal
    mask = col > row
    
    # Load scores into shared memory
    scores = tl.load(scores_ptr + row[:, None] * seq_len + col, mask=mask, other=float('-inf'))
    
    # Compute the maximum score to avoid numerical instability
    max_score = tl.max(scores, axis=1)
    max_score = tl.broadcast_to(max_score[:, None], scores.shape)
    
    # Subtract the max score for numerical stability
    scores = scores - max_score
    
    # Compute the exponential of the scores
    exp_scores = tl.exp(scores)
    
    # Compute the sum of exponentials
    sum_exp_scores = tl.sum(exp_scores, axis=1)
    sum_exp_scores = tl.broadcast_to(sum_exp_scores[:, None], scores.shape)
    
    # Compute the softmax
    softmax = exp_scores / sum_exp_scores
    
    # Store the result back to global memory
    tl.store(output_ptr + row[:, None] * seq_len + col, softmax, mask=mask)

def kernel(scores):
    batch_size, seq_len, _ = scores.shape
    output = torch.empty_like(scores)
    
    # Launch the Triton kernel
    grid = (
        triton.cdiv(seq_len, 32),
        triton.cdiv(seq_len, 32),
        1
    )
    causal_softmax_kernel[grid](
        scores.contiguous(),
        output.contiguous(),
        batch_size,
        seq_len,
        BLOCK_SIZE=32,
        CACHE_LINE_SIZE=128
    )
    
    return output

# Example usage
if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    scores = torch.randn(1, 1024, 1024, dtype=torch.float16, device='cuda')
    reference_output = torch.nn.functional.softmax(scores.float().masked_fill(torch.triu(torch.ones(scores.shape[-2], scores.shape[-1], dtype=torch.bool, device=scores.device), 1), float("-inf")), dim=-1).to(scores.dtype)
    triton_output = kernel(scores)
    
    print("Reference output:", reference_output)
    print("Triton output:", triton_output)
    print("Max diff:", torch.max(torch.abs(reference_output - triton_output)))
