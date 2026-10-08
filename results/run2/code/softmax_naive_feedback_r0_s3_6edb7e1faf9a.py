import torch
import triton
import triton.language as tl

@triton.jit
def row_wise_softmax_kernel(
    X,  # input matrix of shape (M, N)
    Y,  # output matrix of shape (M, N)
    M,  # number of rows
    N,  # number of columns
    BLOCK_SIZE: tl.constexpr,
    **META
):
    # get program ID
    pid = tl.program_id(axis=0)
    # calculate row and column indices
    row = pid // (N // BLOCK_SIZE)
    col = pid % (N // BLOCK_SIZE)
    # load data from global memory to shared memory
    offsets = row * N + col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets, mask=offsets < N, other=-float('inf'))
    # compute max and subtract it for numerical stability
    max_x = tl.max(x, axis=0)
    x = x - max_x
    # compute exp and sum of exp
    exp_x = tl.exp(x)
    sum_exp_x = tl.sum(exp_x, axis=0)
    # compute softmax
    softmax_x = exp_x / sum_exp_x
    # store result back to global memory
    tl.store(Y + offsets, softmax_x, mask=offsets < N)

def kernel(X):
    M, N = X.shape
    # define block size
    BLOCK_SIZE = 128
    # allocate output tensor
    Y = torch.empty_like(X)
    # launch kernel
    grid = lambda META: (triton.cdiv(N, BLOCK_SIZE) * M,)
    row_wise_softmax_kernel[grid](X, Y, M, N, BLOCK_SIZE)
    return Y

# Example usage:
if __name__ == "__main__":
    import torch

    # Create a random 2D tensor of shape (M, N)
    M, N = 1024, 128
    X = torch.rand(M, N, device='cuda', dtype=torch.float16)

    # Compute softmax using the reference implementation
    reference_result = torch.softmax(X.float(), dim=-1).to(X.dtype)

    # Compute softmax using the Triton kernel
    triton_result = kernel(X)

    # Check if the results are close
    assert torch.allclose(reference_result, triton_result, atol=1e-3, rtol=1e-3)
    print("Results match!")
