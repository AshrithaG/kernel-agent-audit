import torch
import triton
import triton.language as tl

@triton.jit
def swiglu_kernel(X, Y, stride_xm, stride_ym, BLOCK_SIZE: tl.constexpr):
    # Get the position of the current thread within the block
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl threadIdx.x
    
    # Load the first half of the vector into registers
    x1 = tl.load(X + idx, mask=(idx < X.shape[0]), other=0.0).to(tl.float32)
    
    # Load the second half of the vector into registers
    x2 = tl.load(X + idx + X.shape[0], mask=(idx < X.shape[0]), other=0.0).to(tl.float32)
    
    # Apply SiLU activation to the first half
    x1 = x1 * tl.sigmoid(x1)
    
    # Multiply the two halves
    y = x1 * x2
    
    # Store the result back to global memory
    tl.store(Y + idx, y, mask=(idx < X.shape[0]))

def kernel(X):
    # Ensure the input tensor is contiguous
    X = X.contiguous()
    
    # Define the output tensor
    Y = torch.empty_like(X)
    
    # Define the grid and block sizes
    BLOCK_SIZE = 128
    grid_size = (triton.cdiv(X.numel(), BLOCK_SIZE),)
    
    # Launch the Triton kernel
    swiglu_kernel[grid_size](X, Y, X.stride(0), Y.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    
    return Y

# Example usage
if __name__ == "__main__":
    import torch.nn.functional as F
    import math

    # Create a random input tensor of shape (batch_size, seq_len, hidden_dim)
    batch_size, seq_len, hidden_dim = 32, 128, 256
    X = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', dtype=torch.float16)

    # Reference implementation
    reference_output = F.silu(X[..., :hidden_dim//2].float()) * X[..., hidden_dim//2:].float()

    # Triton implementation
    triton_output = kernel(X)

    # Check if the outputs are the same within floating-point tolerance
    assert torch.allclose(reference_output, triton_output.float(), atol=1e-3, rtol=1e-3)

    print("Triton implementation is correct!")
