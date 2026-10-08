import torch
import triton
import triton.language as tl
import math

@triton.jit
def mean_kernel(input_ptr, output_ptr, n_elements_per_row, n_rows, BLOCK_SIZE: tl.constexpr):
    # Define the thread index
    row_idx = tl.program_id(0)
    col_idx = tl.load(tl.make_block_ptr(input_ptr, shape=(n_rows, n_elements_per_row), strides=(n_elements_per_row, 1), offsets=(row_idx, 0), block_shape=(BLOCK_SIZE,), order=(1, 0)))

    # Load data from global memory to shared memory
    row_data = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, n_elements_per_row, BLOCK_SIZE):
        mask = col_idx < n_elements_per_row
        row_data += tl.load(tl.make_block_ptr(input_ptr, shape=(n_rows, n_elements_per_row), strides=(n_elements_per_row, 1), offsets=(row_idx, col_idx + i), block_shape=(BLOCK_SIZE,), order=(1, 0)), mask=mask)

    # Compute the sum of elements in the row
    row_sum = tl.sum(row_data)

    # Compute the mean by dividing the sum by the number of elements
    row_mean = row_sum / n_elements_per_row

    # Store the result in global memory
    tl.store(output_ptr + row_idx, row_mean)

def kernel(input_tensor):
    n_rows, n_elements_per_row = input_tensor.shape
    output_tensor = torch.empty(n_rows, dtype=torch.float32, device=input_tensor.device)

    # Define the block size
    BLOCK_SIZE = 128

    # Launch the kernel
    grid = lambda META: (n_rows,)
    mean_kernel[grid](input_tensor.contiguous(), output_tensor, n_elements_per_row, n_rows, BLOCK_SIZE=BLOCK_SIZE)

    return output_tensor.to(input_tensor.dtype)

# Example usage:
if __name__ == "__main__":
    input_tensor = torch.randn((1024, 512), dtype=torch.float16, device='cuda')
    output_tensor = kernel(input_tensor)
    print(output_tensor)
