import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # Calculate the row and column indices of the current thread
    row = pid // (BLOCK_SIZE // 16)
    col = pid % (BLOCK_SIZE // 16)
    
    # Load the element from the input matrix
    x = tl.load(x_ptr + row * BLOCK_SIZE + col, mask=row < n_elements // BLOCK_SIZE, other=0.0)
    
    # Apply ReLU activation
    x = tl.where(x > 0, x, 0.0)
    
    # Store the result back to the output matrix
    tl.store(out_ptr + row * BLOCK_SIZE + col, x, mask=row < n_elements // BLOCK_SIZE)

def kernel(a, b, bias):
    M, K = a.shape
    K, N = b.shape
    assert K == b.shape[0], "Incompatible dimensions for matrix multiplication"
    assert N == bias.shape[0], "Bias dimension mismatch"

    # Compute the matrix multiplication result
    out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    out_matmul = torch.matmul(a.float(), b.float())

    # Add bias
    out_matmul += bias.float()

    # Convert to fp16 if necessary
    if a.dtype == torch.float16:
        out_matmul = out_matmul.half()

    # Define grid size
    BLOCK_SIZE = 128
    grid_size = (M * N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    relu_kernel[grid_size](out_matmul.data_ptr(), out.data_ptr(), M * N, BLOCK_SIZE=BLOCK_SIZE)

    return out

# Example usage
if __name__ == "__main__":
    import torch

    # Create random tensors
    M, K, N = 128, 256, 512
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    bias = torch.randn(N, device='cuda', dtype=torch.float16)

    # Run the custom kernel
    result = kernel(a, b, bias)
    print(result)
