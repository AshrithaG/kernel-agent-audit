import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(a_ptr, b_ptr, bias_ptr, out_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # Define the position of each thread within the block
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Calculate the row and column index of the current thread
    row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create a mask to ensure we do not go out of bounds
    mask = (row < M) & (col < N)
    
    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    
    # Perform the matrix multiplication
    for k in range(0, K, BLOCK_SIZE_K):
        # Load the data from global memory into shared memory
        a_block = tl.load(a_ptr + row[:, None] * K + k[None, :], mask=mask[:, None], other=0.0)
        b_block = tl.load(b_ptr + k[:, None] * N + col[None, :], mask=mask[None, :], other=0.0)
        
        # Perform the dot product
        acc += tl.dot(a_block, b_block)
    
    # Add the bias
    bias_block = tl.load(bias_ptr + col, mask=mask, other=0.0)
    acc += bias_block
    
    # Apply ReLU
    acc = tl.where(acc > 0, acc, 0.0)
    
    # Store the result back to global memory
    tl.store(out_ptr + row[:, None] * N + col[None, :], acc, mask=mask[:, None])

def kernel(a, b, bias):
    M, K = a.shape
    K, N = b.shape
    
    # Compute the grid size
    grid_size = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    
    # Allocate the output tensor
    output = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Launch the kernel
    relu_kernel[grid_size](a, b, bias, output, M=M, N=N, K=K, BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32)
    
    return output

# Example usage
if __name__ == "__main__":
    import torch
    
    # Set device to CUDA
    device = torch.device("cuda")
    
    # Create random tensors
    M, K, N = 1024, 512, 768
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N, device=device, dtype=torch.float16)
    bias = torch.randn(N, device=device, dtype=torch.float16)
    
    # Run the kernel
    result = kernel(a, b, bias)
    
    # Print the result
    print(result)
