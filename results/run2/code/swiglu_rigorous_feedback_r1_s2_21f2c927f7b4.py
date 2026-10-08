import torch
import triton
import triton.language as tl

@triton.jit
def swiglu_kernel(x_ptr, y_ptr, N, H, BLOCK_SIZE: tl.constexpr):
    # Get the program ID for the current thread block and the current thread within the block
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    
    # Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE + bid * H * BLOCK_SIZE
    
    # Generate the offsets for the elements this thread will process
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to ensure we only access valid memory locations
    mask = offsets < N
    
    # Load the input data into shared memory, using the mask to avoid out-of-bounds accesses
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Split the loaded data into two halves
    first_half = x[:H]
    second_half = x[H:]
    
    # Apply the SiLU activation function to the first half
    first_half = first_half * (1 / (1 + tl.exp(-first_half)))
    
    # Multiply the first half by the second half to get the output
    y = first_half * second_half
    
    # Store the result back into the output tensor, using the mask to avoid out-of-bounds writes
    tl.store(y_ptr + offsets, y, mask=mask)

def kernel(x):
    B, N, H = x.shape
    y = torch.empty_like(x)
    
    # Define the grid dimensions for launching the kernel
    grid = lambda opt: (
        triton.cdiv(N, opt.BLOCK_SIZE),
        B,
    )
    
    # Launch the kernel with the specified block size
    swiglu_kernel[grid](x, y, N, H, BLOCK_SIZE=128)
    
    return y

# Example usage:
if __name__ == "__main__":
    x = torch.randn(1024, 2048, device='cuda', dtype=torch.float16)
    y = kernel(x)
    print(y)
