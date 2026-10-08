import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel
@triton.jit
def bias_gelu_kernel(x_ptr, b_ptr, out_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    # Calculate the row and column indices for the current thread
    row = tl.program_id(0)
    col = tl.load(tl.make_block_ptr(b_ptr, shape=(N,), strides=(1,), offsets=(col,), block_shape=(BLOCK_SIZE,), order=(0,)))
    
    # Load the bias value for the current column
    bias = tl.load(tl.make_block_ptr(b_ptr, shape=(N,), strides=(1,), offsets=(col,), block_shape=(BLOCK_SIZE,), order=(0,)))
    
    # Load the input values for the current row
    x = tl.load(tl.make_block_ptr(x_ptr, shape=(M, N), strides=(N, 1), offsets=(row * N, col), block_shape=(BLOCK_SIZE, BLOCK_SIZE), order=(0, 1)))
    
    # Add the bias to the input values
    x += bias
    
    # Apply the tanh approximation of GELU
    cdf = 0.5 * (1.0 + tl.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))
    out = x * cdf
    
    # Store the output values
    tl.store(tl.make_block_ptr(out_ptr, shape=(M, N), strides=(N, 1), offsets=(row * N, col), block_shape=(BLOCK_SIZE, BLOCK_SIZE), order=(0, 1)), out)

# Define the wrapper function to call the Triton kernel
def kernel(x, b):
    M, N = x.shape
    BLOCK_SIZE = 128
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Launch the Triton kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE']),)
    bias_gelu_kernel[grid](x, b, out, M, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

# Example usage
if __name__ == "__main__":
    import torch.nn.functional as F
    
    # Create example inputs
    x = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
    b = torch.randn(512, device='cuda', dtype=torch.float16)
    
    # Run the reference implementation
    ref_output = F.gelu(x + b, approximate="tanh")
    
    # Run the Triton implementation
    triton_output = kernel(x, b)
    
    # Check if the outputs are equal
    print(torch.allclose(ref_output, triton_output))
