import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(A_ptr, B_ptr, Bias_ptr, Out_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
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
    
    # Loop over the K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A and B blocks
        A_block = tl.load(A_ptr + row[:, None] * K + k + tl.arange(0, BLOCK_SIZE_K)[None, :], mask=mask[:, None] & (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < K))
        B_block = tl.load(B_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * N + col[None, :], mask=(k + tl.arange(0, BLOCK_SIZE_K))[:, None] < K & mask[None, :])
        
        # Perform matrix multiplication
        acc += tl.dot(A_block, B_block)
    
    # Add bias
    bias_block = tl.load(Bias_ptr + col, mask=mask)
    acc += bias_block
    
    # Apply ReLU
    acc = tl.where(acc > 0, acc, 0.0)
    
    # Store the result back to global memory
    tl.store(Out_ptr + row[:, None] * N + col[None, :], acc, mask=mask[:, None])

def kernel(A, B, Bias):
    M, K = A.shape
    K, N = B.shape
    
    # Compute the output shape
    output_shape = (M, N)
    
    # Allocate the output tensor
    Output = torch.empty(output_shape, device=A.device, dtype=A.dtype)
    
    # Compute the grid size
    grid_size = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    
    # Launch the kernel
    relu_kernel[grid_size](A, B, Bias, Output, M=M, N=N, K=K, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32)
    
    return Output

# Example usage
if __name__ == "__main__":
    import torch
    
    # Set device to CUDA
    device = torch.device("cuda")
    
    # Create random tensors
    M, K, N = 1024, 512, 768
    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16)
    Bias = torch.randn(N, device=device, dtype=torch.float16)
    
    # Run the kernel
    result = kernel(A, B, Bias)
    
    # Print the result
    print(result)
