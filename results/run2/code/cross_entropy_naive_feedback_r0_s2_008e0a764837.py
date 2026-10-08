import torch
import triton
import triton.language as tl
import math

# Triton kernel for per-row cross-entropy loss
@triton.jit
def cross_entropy_kernel(
    logits_ptr, target_ptr, output_ptr,
    M, V,
    BLOCK_M: tl.constexpr
):
    # Define the row and column indices
    row = tl.program_id(0)
    col = tl.load(target_ptr + row)
    
    # Load the logits for the current row
    logits = tl.load(logits_ptr + row * V + col)
    
    # Calculate the sum of logits for the current row
    sum_logits = tl.zeros([V], dtype=tl.float32)
    for i in range(0, V, BLOCK_M):
        sum_logits += tl.load(logits_ptr + row * V + i)
    
    # Calculate the log of the sum of logits
    log_sum_logits = tl.log(sum_logits)
    
    # Calculate the cross-entropy loss for the current row
    loss = -logits + log_sum_logits
    
    # Store the result in the output array
    tl.store(output_ptr + row, loss)

# Wrapper function to call the Triton kernel
def kernel(logits, target):
    M, V = logits.shape
    output = torch.empty(M, dtype=torch.float32, device=logits.device)
    
    # Define the grid and block sizes
    grid = (M,)
    block = (128,)
    
    # Launch the Triton kernel
    cross_entropy_kernel[grid, block](
        logits.data_ptr(), target.data_ptr(), output.data_ptr(),
        M, V,
        BLOCK_M=block[0]
    )
    
    return output

# Example usage
if __name__ == "__main__":
    logits = torch.rand((1024, 512), dtype=torch.float16, device='cuda')
    target = torch.randint(0, 512, (1024,), dtype=torch.int64, device='cuda')
    
    # Reference implementation
    reference_output = F.cross_entropy(logits.float(), target, reduction="none")
    
    # Triton kernel implementation
    triton_output = kernel(logits, target)
    
    # Check if the outputs are the same within floating-point tolerance
    assert torch.allclose(reference_output, triton_output, atol=1e-5, rtol=1e-5)
    
    print("Outputs match within tolerance.")
