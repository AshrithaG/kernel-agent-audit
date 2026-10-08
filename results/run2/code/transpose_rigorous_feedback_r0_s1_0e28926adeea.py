import torch
import triton
import triton.language as tl

@triton.jit
def transpose_kernel(
    x_ptr, y_ptr, M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Calculate the row and column indices for this thread block
    row = pid // num_pid_n
    col = pid % num_pid_n

    # Load the input data into shared memory
    x_block = tl.load(x_ptr + row * BLOCK_M * N + col * BLOCK_N, mask=(row < num_pid_m) & (col < num_pid_n))

    # Transpose the block
    y_block = tl.trans(x_block)

    # Store the transposed block to the output
    tl.store(y_ptr + col * BLOCK_N * M + row * BLOCK_M, y_block, mask=(row < num_pid_m) & (col < num_pid_n))

def kernel(x):
    M, N = x.shape
    BLOCK_M = 32
    BLOCK_N = 32

    # Allocate output tensor
    y = torch.empty_like(x)

    # Launch the Triton kernel
    grid = lambda META: (tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N),)
    transpose_kernel[grid](x.contiguous(), y.contiguous(), M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    return y

# Example usage
if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    
    start_time = time.time()
    y = kernel(x)
    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time:.6f} seconds")
    print("Transpose successful:", torch.allclose(y, x.t()))
