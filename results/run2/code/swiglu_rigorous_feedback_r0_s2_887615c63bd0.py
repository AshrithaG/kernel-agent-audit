import torch
import triton
import triton.language as tl

# Define the Triton kernel for SwiGLU
@triton.jit
def swiglu_kernel(
    X, Y,
    X_ptr, Y_ptr,
    N, M,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get the thread ID and block ID
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    
    # Calculate the row and column indices
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = bid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    
    # Load the input data
    x = tl.load(X_ptr + row[:, None] * M + col[None, :], mask=row[:, None] < N, other=0.0)
    
    # Split the input into two halves
    half = M // 2
    x1 = x[:, :half]
    x2 = x[:, half:]
    
    # Apply SiLU to the first half
    x1_silu = x1 * tl.sigmoid(x1)
    
    # Multiply the SiLU result with the second half
    y = x1_silu * x2
    
    # Store the output data
    tl.store(Y_ptr + row[:, None] * M + col[None, :], y, mask=row[:, None] < N)

# Define the kernel wrapper function
def kernel(*inputs):
    # Unpack the inputs
    X = inputs[0]
    
    # Get the shape of the input tensor
    N, M = X.shape
    
    # Define the block and group sizes
    BLOCK_SIZE = 128
    GROUP_SIZE = 32
    
    # Create the output tensor
    Y = torch.empty_like(X)
    
    # Launch the Triton kernel
    grid = lambda opt: (
        triton.cdiv(N, BLOCK_SIZE),
        triton.cdiv(M, GROUP_SIZE),
        1
    )
    swiglu_kernel[grid](
        X, Y,
        X.data_ptr(), Y.data_ptr(),
        N, M,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE,
        num_warps=4
    )
    
    return Y

# Example usage
if __name__ == "__main__":
    # Create a random input tensor
    X = torch.randn((1024, 512), device='cuda', dtype=torch.float16)
    
    # Apply the kernel
    Y = kernel(X)
    
    # Print the shape and dtype of the output tensor
    print("Output shape:", Y.shape)
    print("Output dtype:", Y.dtype)
