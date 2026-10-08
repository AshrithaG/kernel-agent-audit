import torch
import triton
import triton.language as tl
import math

# Define the Triton kernel
@triton.jit
def bias_gelu_kernel(
    X_ptr, B_ptr, Y_ptr,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # Compute the indices
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    block_id_m = pid // num_pid_n
    block_id_n = pid % num_pid_n

    # Calculate the row and column indices
    row = block_id_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col = block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Load the data from global memory
    X = tl.load(X_ptr[row[:, None] * N + col], mask=row[:, None] < M & col < N, other=0.0)
    B = tl.load(B_ptr[col], mask=col < N, other=0.0)

    # Add the bias
    X += B

    # Tanh approximation of GELU
    CDF = 0.5 * (1.0 + tl.tanh((math.sqrt(2.0 / math.pi) * (X + 0.044715 * (X ** 3)))))

    # Store the result back to global memory
    tl.store(Y_ptr[row[:, None] * N + col], CDF, mask=row[:, None] < M & col < N)

# Define the wrapper function
def kernel(X, B):
    # Ensure the inputs are on the same device
    assert X.device == B.device, "Inputs must be on the same device"
    
    # Get the shape of the input tensors
    M, N = X.shape
    
    # Define the block size
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    
    # Allocate the output tensor
    Y = torch.empty_like(X)
    
    # Launch the Triton kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) *
        triton.cdiv(N, META['BLOCK_SIZE_N'])
    )
    bias_gelu_kernel[grid](
        X.data_ptr(), B.data_ptr(), Y.data_ptr(),
        M, N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=4
    )
    
    return Y

# Example usage
if __name__ == "__main__":
    # Create example inputs
    X = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
    B = torch.randn(512, device='cuda', dtype=torch.float16)
    
    # Run the Triton kernel
    Y = kernel(X, B)
    
    # Print the result
    print(Y)
