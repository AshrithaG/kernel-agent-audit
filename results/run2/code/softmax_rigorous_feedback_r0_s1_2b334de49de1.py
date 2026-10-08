import torch
import triton
import triton.language as tl

@triton.jit
def rowwise_softmax_kernel(X, Y, BLOCK_SIZE: tl.constexpr):
    # Get the position of the element within the block
    pid = tl.program_id(axis=0)
    x_offset = pid * BLOCK_SIZE
    cols = tl.num_programs(axis=0)

    # Load the input data into shared memory
    x = tl.load(X + x_offset, mask=x_offset < cols, other=-float('inf'))

    # Compute the max value of each row
    max_x = tl.max(x, axis=1)
    max_x = tl.broadcast_to(max_x, x.shape)

    # Subtract the max value from each element to avoid overflow
    x = x - max_x

    # Compute the exponential of each element
    exp_x = tl.exp(x)

    # Compute the sum of the exponentials for each row
    sum_exp_x = tl.sum(exp_x, axis=1)
    sum_exp_x = tl.broadcast_to(sum_exp_x, x.shape)

    # Compute the softmax
    softmax_x = exp_x / sum_exp_x

    # Store the result back to global memory
    tl.store(Y + x_offset, softmax_x, mask=x_offset < cols)

def kernel(X):
    # Get the shape of the input tensor
    rows, cols = X.shape

    # Define the block size
    BLOCK_SIZE = triton.next_power_of_2(cols)

    # Allocate output tensor
    Y = torch.empty_like(X)

    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(rows, BLOCK_SIZE),)
    rowwise_softmax_kernel[grid](X, Y, BLOCK_SIZE=BLOCK_SIZE)

    return Y
