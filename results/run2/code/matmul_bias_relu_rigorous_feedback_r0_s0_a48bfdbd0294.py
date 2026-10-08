import torch
import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def relu_kernel(a_ptr, b_ptr, bias_ptr, out_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Define the indices for this thread
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid = num_pid_m * num_pid_n
    block_id = pid // num_pid
    block_row = block_id // num_pid_n
    block_col = block_id % num_pid_n
    row_within_block = pid % num_pid
    row_within_block_m = row_within_block // BLOCK_SIZE_N
    row_within_block_n = row_within_block % BLOCK_SIZE_N
    row = block_row * BLOCK_SIZE_M + row_within_block_m
    col = block_col * BLOCK_SIZE_N + row_within_block_n

    # Load data from global memory to shared memory
    a_shared = tl.zeros((BLOCK_SIZE_M, K), dtype=tl.float32)
    b_shared = tl.zeros((K, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(K):
        a_shared[row_within_block_m, k] = tl.load(a_ptr + row * K + k, mask=row < M, other=0.0)
        b_shared[k, row_within_block_n] = tl.load(b_ptr + k * N + col, mask=col < N, other=0.0)

    # Compute the dot product
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(K):
        acc += a_shared[row_within_block_m, k] * b_shared[k, row_within_block_n]

    # Add bias and apply ReLU
    if row < M and col < N:
        acc[row_within_block_m, row_within_block_n] = tl.maximum(acc[row_within_block_m, row_within_block_n] + tl.load(bias_ptr + col), 0.0)
        tl.store(out_ptr + row * N + col, acc[row_within_block_m, row_within_block_n])

# Define the kernel wrapper
def kernel(a, b, bias):
    M, K = a.shape
    _, N = b.shape
    assert a.dtype == torch.float16 and b.dtype == torch.float16 and bias.dtype == torch.float16
    out = torch.empty_like(a.matmul(b) + bias)

    # Launch the kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    triton.jit.kernel[grid](relu_kernel, a.contiguous(), b.contiguous(), bias.contiguous(), out.contiguous(), M, N, K, BLOCK_SIZE_M=16, BLOCK_SIZE_N=16)

    return out

# Example usage
if __name__ == "__main__":
    import time

    # Create random inputs
    M, K, N = 1024, 512, 768
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    bias = torch.randn(N, device='cuda', dtype=torch.float16)

    # Measure the time taken by the reference implementation
    start_time = time.time()
    ref_out = torch.relu(a.float() @ b.float() + bias.float()).to(a.dtype)
    ref_time = time.time() - start_time
    print(f"Reference time: {ref_time:.6f} seconds")

    # Measure the time taken by the Triton kernel
    start_time = time.time()
    triton_out = kernel(a, b, bias)
    triton_time = time.time() - start_time
    print(f"Triton time: {triton_time:.6f} seconds")

    # Check the correctness of the output
    assert torch.allclose(ref_out, triton_out, atol=1e-3, rtol=1e-3)
    print("Output is correct.")
