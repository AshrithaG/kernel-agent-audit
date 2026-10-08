import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, y_ptr, stride_x, stride_y, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Define the position of each thread within the block
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Calculate the row and column index of the current thread
    row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create a mask to ensure we do not go out of bounds
    mask = (row < M) & (col < N)
    
    # Load the data from global memory into shared memory
    x = tl.load(x_ptr + row[:, None] * stride_x + col[None, :], mask=mask[:, None])
    
    # Apply ReLU
    x = tl.where(x > 0, x, 0.0)
    
    # Store the result back to global memory
    tl.store(y_ptr + row[:, None] * stride_y + col[None, :], x, mask=mask[:, None])

def kernel(a, b, bias):
    M, K = a.shape
    K, N = b.shape
    
    # Compute the output shape
    output_shape = (M, N)
    
    # Allocate the output tensor
    output = torch.empty(output_shape, device=a.device, dtype=a.dtype)
    
    # Compute the grid size
    grid_size = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    
    # Launch the kernel
    relu_kernel[grid_size](a, b, bias, output, M=M, N=N, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128)
    
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
