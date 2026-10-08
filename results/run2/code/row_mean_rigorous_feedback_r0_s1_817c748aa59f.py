import torch
import triton
import triton.language as tl

# Define the Triton kernel for mean computation
@triton.jit
def mean_kernel(
    X,  # input matrix
    Y,  # output vector
    N,  # number of rows
    M,  # number of columns
    BLOCK_SIZE: tl.constexpr,  # block size
):
    pid = tl.program_id(axis=0)  # get the ID of the current program instance
    stride_pid = BLOCK_SIZE * BLOCK_SIZE  # stride for each program instance
    offsets = tl.arange(0, BLOCK_SIZE) + pid * stride_pid  # calculate offsets
    mask = offsets < M  # create a mask to handle cases where M is not a multiple of BLOCK_SIZE
    x = tl.load(X + offsets, mask=mask, other=0.0)  # load data from X, using mask to avoid out-of-bounds access
    sum_x = tl.sum(x, axis=0)  # sum the elements along the block
    tl.store(Y + pid, sum_x / M)  # store the result in Y, dividing by M to get the mean

# Define the wrapper function to call the Triton kernel
def kernel(X):
    N, M = X.shape  # get the shape of the input tensor
    Y = torch.empty(N, dtype=torch.float32, device=X.device)  # create an empty tensor for the output
    BLOCK_SIZE = 128  # define the block size for the Triton kernel
    grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)  # calculate the grid size
    mean_kernel[grid](X, Y, N, M, BLOCK_SIZE=BLOCK_SIZE)  # launch the Triton kernel
    return Y.to(X.dtype)  # convert the output to the same dtype as the input

# Example usage
if __name__ == "__main__":
    X = torch.randn((1024, 512), dtype=torch.float16, device='cuda')
    Y = kernel(X)
    print(Y)
