import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel
@triton.jit
def bias_gelu_kernel(
    X, B, Y,
    X_row_stride, X_col_stride,
    B_stride,
    Y_row_stride, Y_col_stride,
    BLOCK_SIZE: tl.constexpr
):
    # Get the row index and column index
    row = tl.program_id(0)
    col = tl.load(B + col)  # Load bias value for this column
    x = tl.load(X + row * X_row_stride + col * X_col_stride, mask=col < X_col_stride)
    
    # Add bias
    x += col
    
    # Approximate GELU using tanh
    k = 0.7978845608028654 * x
    y = 0.5 * x * (1.0 + tl.tanh(k))
    
    # Store the result
    tl.store(Y + row * Y_row_stride + col * Y_col_stride, y, mask=col < X_col_stride)

# Define the wrapper function
def kernel(X, B):
    # Ensure inputs are on the same device
    assert X.device == B.device, "Inputs must be on the same device"
    assert X.dtype == torch.float16, "Input X must be of type torch.float16"
    assert B.dtype == torch.float16, "Input B must be of type torch.float16"
    
    # Get the shape of the inputs
    M, N = X.shape
    
    # Allocate output tensor
    Y = torch.empty_like(X)
    
    # Define grid and block dimensions
    grid = lambda META: (M, 1)
    block = lambda META: (META['BLOCK_SIZE'],)
    
    # Launch the Triton kernel
    bias_gelu_kernel[grid](X, B, Y,
                          X.stride(0), X.stride(1),
                          B.stride(0),
                          Y.stride(0), Y.stride(1),
                          BLOCK_SIZE=128)
    
    return Y

# Example usage
if __name__ == "__main__":
    # Create example inputs
    X = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
    B = torch.randn(512, device='cuda', dtype=torch.float16)
    
    # Run the kernel
    Y = kernel(X, B)
    
    # Print the result
    print(Y)
