import torch
import triton
import triton.language as tl

# Define the Triton kernel for transposing a 2D tensor
@triton.jit
def transpose_kernel(
    X, Y,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    stride_xm: tl.constexpr, stride_xn: tl.constexpr,
    stride_ym: tl.constexpr, stride_yn: tl.constexpr
):
    # Get the program ID for the block
    pid = tl.program_id(axis=0)
    # Calculate the row and column indices for the block
    row = (pid // (N // BLOCK_N)) * BLOCK_M
    col = (pid % (N // BLOCK_N)) * BLOCK_N
    # Load the block from the input tensor
    x = tl.load(X + row * stride_xm + col * stride_xn, mask=(row < M) & (col < N), other=0.0)
    # Transpose the block
    y = tl.trans(x)
    # Store the transposed block to the output tensor
    tl.store(Y + col * stride_ym + row * stride_yn, y, mask=(row < M) & (col < N))

# Function to transpose a 2D tensor using the Triton kernel
def kernel(X):
    M, N = X.shape
    # Define the block size
    BLOCK_M = 32
    BLOCK_N = 32
    # Calculate the number of blocks needed
    num_blocks = (M * N + BLOCK_M * BLOCK_N - 1) // (BLOCK_M * BLOCK_N)
    # Create an output tensor with the transposed shape
    Y = torch.empty((N, M), dtype=torch.float16, device=X.device)
    # Launch the Triton kernel
    transpose_kernel[triton.next_power_of_2(num_blocks)](
        X, Y,
        M=M, N=N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        stride_xm=X.stride(0), stride_xn=X.stride(1),
        stride_ym=Y.stride(0), stride_yn=Y.stride(1)
    )
    return Y.contiguous()

# Example usage
if __name__ == "__main__":
    # Create a random 2D tensor
    X = torch.randn((1024, 2048), dtype=torch.float16, device='cuda')
    # Transpose the tensor using the Triton kernel
    Y = kernel(X)
    # Verify the result
    reference_Y = X.t().contiguous()
    assert torch.allclose(Y, reference_Y, atol=1e-3)
    print("Transpose operation verified successfully!")
