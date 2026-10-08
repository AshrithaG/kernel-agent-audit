import torch
import triton
import triton.language as tl

@triton.jit
def transpose_kernel(input_ptr, output_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    
    # Calculate the row and column indices for the current thread
    row = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    col = bid * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Load the input values into registers
    input_values = tl.load(input_ptr + col[:, None] * M + row[None, :], mask=row[:, None] < M & col[None, :] < N, other=0.0)
    
    # Store the transposed values into the output buffer
    tl.store(output_ptr + row[:, None] * N + col[None, :], input_values, mask=row[:, None] < M & col[None, :] < N)

def kernel(input_tensor):
    M, N = input_tensor.shape
    BLOCK_M = 16
    BLOCK_N = 16
    
    # Allocate output tensor
    output_tensor = torch.empty((N, M), dtype=torch.float16, device=input_tensor.device)
    
    # Launch the Triton kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )
    transpose_kernel[grid](input_tensor, output_tensor, M, N, BLOCK_M, BLOCK_N)
    
    return output_tensor

# Example usage
if __name__ == "__main__":
    import numpy as np
    torch.manual_seed(0)
    input_tensor = torch.randn(128, 256, dtype=torch.float16, device='cuda')
    
    # Reference implementation
    reference_output = input_tensor.t().contiguous()
    
    # Triton kernel implementation
    triton_output = kernel(input_tensor)
    
    # Verify correctness
    assert torch.allclose(reference_output, triton_output, atol=1e-3, rtol=1e-3)
    print("Transposition is correct!")
    
    # Print performance
    import time
    start_time = time.time()
    for _ in range(100):
        kernel(input_tensor)
    end_time = time.time()
    print(f"Triton kernel execution time: {end_time - start_time:.4f} seconds")
