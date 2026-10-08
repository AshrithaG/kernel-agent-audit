import torch
import triton
import triton.language as tl

@triton.jit
def rowwise_softmax_kernel(
    X,  # input matrix
    Y,  # output matrix
    maxes,  # max values per row
    sums,  # sum of exponentials per row
    N,  # number of rows
    M,  # number of columns
    BLOCK_SIZE: tl.constexpr,
):
    # get the row index
    row = tl.program_id(0)
    # get the column index
    col = tl.load(tl.arange(0, BLOCK_SIZE), mask=col < M, other=0)
    # load the input value
    x = tl.load(X + row * M + col, mask=col < M, other=0)
    # compute the max value per row
    max_value = tl.max(x, axis=1)
    # store the max value per row
    tl.store(maxes + row, max_value)
    # subtract the max value from the input to avoid overflow
    x = x - max_value
    # compute the exponential
    exp_x = tl.exp(x)
    # store the sum of exponentials per row
    tl.atomic_add(sums + row, exp_x)
    # compute the softmax
    softmax_x = exp_x / sums[row]
    # store the result
    tl.store(Y + row * M + col, softmax_x, mask=col < M)

def kernel(X):
    N, M = X.shape
    assert X.dtype == torch.float16, "Input must be of type float16"
    assert X.is_cuda, "Input must be on the GPU"
    
    # allocate output and temporary storage tensors
    Y = torch.empty_like(X)
    maxes = torch.empty((N,), device=X.device, dtype=torch.float32)
    sums = torch.empty((N,), device=X.device, dtype=torch.float32)
    
    # define grid and block dimensions
    BLOCK_SIZE = 128
    grid = lambda: (N,)
    block = lambda: (BLOCK_SIZE,)
    
    # launch the Triton kernel
    rowwise_softmax_kernel[grid, block](
        X, Y, maxes, sums, N, M, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return Y

# Example usage:
# X = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
# Y = kernel(X)
# print(Y)
