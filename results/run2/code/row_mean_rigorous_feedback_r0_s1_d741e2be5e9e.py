import torch
import triton
import triton.language as tl

@triton.jit
def mean_kernel(
    X,  # pointer to input matrix
    Y,  # pointer to output matrix
    stride_xm, stride_xn,  # the strides of the input matrix (x, y)
    stride_ym, stride_yn,  # the strides of the output matrix (x, y)
    N,  # the number of columns in the input matrix
    BLOCK_SIZE: tl.constexpr,  # the size of the block to use for reduction
):
    # Each thread computes the sum of a block of elements
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    # Load a block of elements from the input matrix
    block_start = row_idx * stride_xm + col_idx * BLOCK_SIZE * stride_xn
    block_end = block_start + BLOCK_SIZE * stride_xn
    block_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(block_start, block_end, stride_xn):
        block_sum += tl.load(X + i, mask=i < N * stride_xn, other=0.0)

    # Reduce the block sum to a single value
    block_sum = tl.sum(block_sum, axis=0)

    # Store the result in the output matrix
    if col_idx == 0:
        tl.store(Y + row_idx * stride_ym, block_sum / N, mask=row_idx < N)

def kernel(X):
    M, N = X.shape
    Y = torch.empty((M,), dtype=torch.float32, device=X.device)
    grid = (triton.cdiv(M, 1), triton.cdiv(N, 128))
    mean_kernel[grid](
        X, Y,
        X.stride(0), X.stride(1),
        Y.stride(0), Y.stride(1),
        N,
        BLOCK_SIZE=128,
    )
    return Y.to(X.dtype)
