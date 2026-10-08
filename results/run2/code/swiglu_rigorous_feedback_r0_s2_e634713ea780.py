import torch
import triton
import triton.language as tl

# Define the Triton kernel for SwiGLU
@triton.jit
def swiglu_kernel(
    X, Y,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the position within the block
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    
    # Calculate the start index for this block
    x_start = pid * BLOCK_SIZE + bid * N
    
    # Load the first half and second half of the vector
    x_first_half = tl.load(X + x_start, mask=x_start < N, other=0.0)
    x_second_half = tl.load(X + x_start + N // 2, mask=x_start + N // 2 < N, other=0.0)
    
    # Apply SiLU activation to the first half
    x_first_half_silu = x_first_half * tl.sigmoid(x_first_half)
    
    # Multiply the SiLU result with the second half
    y = x_first_half_silu * x_second_half
    
    # Store the result
    tl.store(Y + x_start, y, mask=x_start < N)

# Define the kernel wrapper function
def kernel(*inputs):
    X = inputs[0]
    N = X.shape[-1]
    BLOCK_SIZE = 128
    
    # Create output tensor
    Y = torch.empty_like(X)
    
    # Launch the Triton kernel
    grid = lambda META: (
        triton.cdiv(N, META['BLOCK_SIZE']),
        X.shape[0],
    )
    swiglu_kernel[grid](X, Y, N=N, BLOCK_SIZE=BLOCK_SIZE)
    
    return Y

# Example usage
if __name__ == "__main__":
    # Create a random float16 tensor
    X = torch.rand((32, 1024), device='cuda', dtype=torch.float16)
    
    # Run the kernel
    Y = kernel(X)
    
    print(Y)
