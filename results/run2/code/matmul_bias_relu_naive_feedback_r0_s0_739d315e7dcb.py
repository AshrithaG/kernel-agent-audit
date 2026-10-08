import torch
import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def matmul_bias_relu_kernel(a_ptr, b_ptr, bias_ptr, out_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # Define the thread indices
    m = tl.program_id(0)
    n = tl.program_id(1)

    # Each thread computes one element of the output matrix
    pid = tl.pid()
    cols_per_pid = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    col_offset = pid % cols_per_pid * BLOCK_SIZE_N
    row_offset = pid // cols_per_pid * BLOCK_SIZE_M

    # Load the bias into shared memory
    bias_block = tl.load(bias_ptr + n * BLOCK_SIZE_N, mask=n < N, other=0.0)

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Perform the matrix multiplication
    for k in range(K // BLOCK_SIZE_K):
        # Load the blocks of A and B into shared memory
        a_block = tl.load(a_ptr + m * BLOCK_SIZE_M * K + k * BLOCK_SIZE_K, mask=m < M, other=0.0)
        b_block = tl.load(b_ptr + n * BLOCK_SIZE_N * K + k * BLOCK_SIZE_K, mask=n < N, other=0.0)

        # Compute the dot product
        acc += tl.dot(a_block, b_block)

    # Add the bias
    acc += bias_block

    # Apply ReLU
    acc = tl.maximum(acc, 0.0)

    # Store the result
    out_ptr = out_ptr + m * BLOCK_SIZE_M * N + n * BLOCK_SIZE_N
    tl.store(out_ptr, acc, mask=m < M & n < N)

# Define the Triton wrapper function
def matmul_bias_relu(a, b, bias):
    # Get the shapes of the input tensors
    M, K = a.shape
    K, N = b.shape

    # Define the grid size
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    num_blocks_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

    # Allocate the output tensor
    out = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Launch the Triton kernel
    matmul_bias_relu_kernel[(num_blocks_m, num_blocks_n)](
        a.contiguous().cuda().half(), 
        b.contiguous().cuda().half(), 
        bias.contiguous().cuda().half(), 
        out.contiguous().cuda().half(),
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M, 
        BLOCK_SIZE_N=BLOCK_SIZE_N, 
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    return out

# Example usage
if __name__ == "__main__":
    import torch
    import triton
    import triton.language as tl

    # Create example input tensors
    a = torch.rand((1024, 512), device='cuda', dtype=torch.float16)
    b = torch.rand((512, 256), device='cuda', dtype=torch.float16)
    bias = torch.rand((256,), device='cuda', dtype=torch.float16)

    # Run the reference implementation
    ref_output = torch.relu(a.float() @ b.float() + bias.float()).to(a.dtype)

    # Run the Triton implementation
    triton_output = matmul_bias_relu(a, b, bias)

    # Check if the outputs are the same
    assert torch.allclose(ref_output, triton_output, atol=1e-3, rtol=1e-3)

    print("Outputs match!")
