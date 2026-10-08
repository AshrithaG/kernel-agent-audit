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
    # Define the program ID
    pid = tl.program_id(axis=0)
    
    # Compute the row and column indices
    row = tl.arange(0, BLOCK_M)
    col = tl.arange(0, BLOCK_N)
    
    # Compute the global row and column indices
    rm = pid // (N // BLOCK_N) * BLOCK_M + row
    rn = pid % (N // BLOCK_N) * BLOCK_N + col
    
    # Load the data from the input tensor
    x = tl.load(X + rm * stride_xm + rn * stride_xn, mask=(rm < M) & (rn < N))
    
    # Store the data to the output tensor
    tl.store(Y + rn * stride_ym + rm * stride_yn, x, mask=(rm < M) & (rn < N))

# Define the kernel function
def kernel(X):
    M, N = X.shape
    # Define the block size
    BLOCK_M = 32
    BLOCK_N = 32
    
    # Allocate the output tensor
    Y = torch.empty_like(X)
    
    # Launch the Triton kernel
    grid = lambda META: (triton.cdiv(N, BLOCK_N),)
    transpose_kernel[grid](
        X, Y,
        M=M, N=N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        stride_xm=X.stride(0), stride_xn=X.stride(1),
        stride_ym=Y.stride(0), stride_yn=Y.stride(1)
    )
    
    return Y.contiguous()

# Example usage
if __name__ == "__main__":
    import torch
    import time
    
    # Create a random 2D tensor
    X = torch.rand((1024, 2048), device='cuda', dtype=torch.float16)
    
    # Measure the time taken by the reference implementation
    start_time = time.time()
    Y_ref = X.t().contiguous()
    ref_time = time.time() - start_time
    print(f"Reference time: {ref_time:.4f} seconds")
    
    # Measure the time taken by the Triton kernel
    start_time = time.time()
    Y_triton = kernel(X)
    triton_time = time.time() - start_time
    print(f"Triton time: {triton_time:.4f} seconds")
    
    # Check if the results are the same
    assert torch.allclose(Y_ref, Y_triton)
