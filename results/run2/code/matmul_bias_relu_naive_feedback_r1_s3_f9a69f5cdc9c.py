import torch
import triton
import triton.language as tl

@triton.jit
def matmul_bias_relu_kernel(a_ptr, b_ptr, bias_ptr, output_ptr, M, N, K, 
                            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # Define the position of each thread within the block
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Calculate the row and column index of the current thread
    row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create a mask to ensure we do not go out of bounds
    mask = (row < M) & (col < N)
    
    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Perform the matrix multiplication
    for k in range(0, K, BLOCK_SIZE_K):
        # Load the data from global memory into shared memory
        a_block = tl.load(a_ptr + row[:, None] * K + (k + tl.arange(0, BLOCK_SIZE_K))[None, :], mask=mask[:, None], other=0.0)
        b_block = tl.load(b_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * N + col[None, :], mask=(k + tl.arange(0, BLOCK_SIZE_K))[:, None] < K & mask[None, :], other=0.0)
        
        # Perform the matrix multiplication
        acc += tl.dot(a_block, b_block)
    
    # Add the bias
    bias_block = tl.load(bias_ptr + col, mask=col < N)
    acc += bias_block
    
    # Apply ReLU
    acc = tl.where(acc > 0, acc, 0.0)
    
    # Store the result back to global memory
    tl.store(output_ptr + row[:, None] * N + col[None, :], acc, mask=mask[:, None])

def kernel(a, b, bias):
    M, K = a.shape
    K, N = b.shape
    
    # Compute the grid size
    grid_size = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    
    # Allocate the output tensor
    output = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # Launch the kernel
    matmul_bias_relu_kernel[grid_size](a, b, bias, output, M=M, N=N, K=K, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128)
    
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
