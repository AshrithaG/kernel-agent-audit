import torch
import triton
import triton.language as tl

@triton.jit
def transpose_kernel(input_ptr, output_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Calculate the row and column index in the block
    row = tl.arange(0, BLOCK_M)
    col = tl.arange(0, BLOCK_N)
    
    # Calculate the global row and column index
    row_idx = row + (pid // num_pid_n) * BLOCK_M
    col_idx = col + (pid % num_pid_n) * BLOCK_N
    
    # Load data from the input tensor
    input_block = tl.load(input_ptr + row_idx[:, None] * N + col_idx[None, :], mask=row_idx[:, None] < M & col_idx[None, :] < N, other=0.0)
    
    # Transpose the block
    output_block = tl.transpose(input_block)
    
    # Store the transposed block to the output tensor
    tl.store(output_ptr + col_idx[:, None] * M + row_idx[None, :], output_block, mask=col_idx[:, None] < N & row_idx[None, :] < M)

def kernel(x):
    M, N = x.shape
    BLOCK_M = 128
    BLOCK_N = 128
    
    # Allocate output tensor
    y = torch.empty((N, M), dtype=x.dtype, device=x.device)
    
    # Launch the Triton kernel
    grid = lambda META: (tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N),)
    transpose_kernel[grid](x, y, M, N, BLOCK_M, BLOCK_N)
    
    return y

# Example usage
if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    x = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
    y = kernel(x)
    print(y.shape)  # Should be (512, 1024)
    print(y.is_contiguous())  # Should be True
