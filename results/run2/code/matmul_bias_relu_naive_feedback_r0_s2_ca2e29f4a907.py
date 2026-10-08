import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, y_ptr, rows, cols, BLOCK_SIZE: tl.constexpr):
    # Compute the row and column indices for this thread
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Load the value from the input matrix
    if row < rows and col < cols:
        x_val = tl.load(x_ptr + row * cols + col)
        # Apply ReLU
        y_val = tl.where(x_val > 0, x_val, 0)
        # Store the result in the output matrix
        tl.store(y_ptr + row * cols + col, y_val)

def kernel(a, b, bias):
    M, K = a.shape
    K, N = b.shape
    assert K == bias.shape[0], "Bias dimension mismatch"
    
    # Perform matrix multiplication
    c = torch.matmul(a.float(), b.float())
    
    # Add bias
    c = c + bias.float()
    
    # Allocate output tensor
    y = torch.empty_like(c)
    
    # Define grid and block sizes
    BLOCK_SIZE = 32
    grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
    block = (BLOCK_SIZE, BLOCK_SIZE)
    
    # Launch the Triton kernel
    relu_kernel[grid, block](c.data_ptr(), y.data_ptr(), M, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return y.to(a.dtype)

# Example usage
if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    a = torch.randn(128, 256, device='cuda', dtype=torch.float16)
    b = torch.randn(256, 512, device='cuda', dtype=torch.float16)
    bias = torch.randn(512, device='cuda', dtype=torch.float16)
    
    reference_output = torch.relu(a.float() @ b.float() + bias.float()).to(a.dtype)
    triton_output = kernel(a, b, bias)
    
    print("Reference output:", reference_output)
    print("Triton output:", triton_output)
    print("Are outputs equal?", torch.allclose(reference_output, triton_output))
