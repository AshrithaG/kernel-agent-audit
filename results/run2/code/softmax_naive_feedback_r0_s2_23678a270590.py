import torch
import triton
import triton.language as tl
import math

@triton.jit
def softmax_kernel(input_ptr, output_ptr, stride, n_elements, BLOCK_SIZE: tl.constexpr):
    # Load input data into shared memory
    row_id = tl.program_id(0)
    col_id = tl.load(input_ptr + row_id * stride + tl.arange(0, BLOCK_SIZE))
    
    # Compute max value to avoid numerical instability
    max_val = tl.max(col_id, axis=0)
    max_val = tl.broadcast_to(max_val, col_id.shape)
    
    # Subtract max value from each element
    col_id = col_id - max_val
    
    # Compute exponentials
    exp_col_id = tl.exp(col_id)
    
    # Compute sum of exponentials
    sum_exp = tl.sum(exp_col_id, axis=0)
    
    # Compute softmax
    softmax_col_id = exp_col_id / sum_exp
    
    # Store result back to global memory
    tl.store(output_ptr + row_id * stride + tl.arange(0, BLOCK_SIZE), softmax_col_id)

def kernel(input_tensor):
    # Ensure input is a float16 tensor
    if input_tensor.dtype != torch.float16:
        raise ValueError("Input tensor must be of type float16")
    
    # Get input dimensions
    batch_size, num_features = input_tensor.shape
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Allocate output tensor
    output_tensor = torch.empty_like(input_tensor)
    
    # Define grid and block sizes
    grid = lambda opt: (batch_size, )
    block = (BLOCK_SIZE, )
    
    # Launch kernel
    softmax_kernel[grid](input_tensor.data_ptr(), output_tensor.data_ptr(), num_features, batch_size * num_features, BLOCK_SIZE)
    
    return output_tensor

# Example usage
if __name__ == "__main__":
    import time
    import torch
    
    # Create a random float16 tensor
    input_tensor = torch.rand((1024, 1024), dtype=torch.float16, device='cuda')
    
    # Measure time for reference implementation
    start_time = time.time()
    reference_output = torch.softmax(input_tensor.float(), dim=-1).to(torch.float16)
    reference_time = time.time() - start_time
    print(f"Reference time: {reference_time:.6f} seconds")
    
    # Measure time for Triton implementation
    start_time = time.time()
    triton_output = kernel(input_tensor)
    triton_time = time.time() - start_time
    print(f"Triton time: {triton_time:.6f} seconds")
    
    # Check correctness
    assert torch.allclose(reference_output, triton_output, atol=1e-3, rtol=1e-3), "Outputs do not match"
    print("Output is correct!")
