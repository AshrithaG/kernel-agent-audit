import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride_x, stride_y, N, M, BLOCK_SIZE: tl.constexpr):
    # Load data into registers
    row_id = tl.program_id(0)
    col_id = tl.arange(0, BLOCK_SIZE)
    mask = col_id < M

    x = tl.load(x_ptr + row_id * stride_x + col_id, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    max_val = tl.max(x, axis=1)
    max_val = tl.broadcast_to(max_val[:, None], (BLOCK_SIZE,))
    x = x - max_val
    
    # Compute exp
    exp_x = tl.exp(x)
    
    # Compute sum of exps
    sum_exp_x = tl.sum(exp_x, axis=1)
    sum_exp_x = tl.broadcast_to(sum_exp_x[:, None], (BLOCK_SIZE,))
    
    # Compute softmax
    softmax_x = exp_x / sum_exp_x
    
    # Store results
    tl.store(y_ptr + row_id * stride_y + col_id, softmax_x, mask=mask)

def kernel(input_tensor):
    N, M = input_tensor.shape
    output_tensor = torch.empty_like(input_tensor, dtype=torch.float32)
    
    # Define grid and block dimensions
    BLOCK_SIZE = 128
    grid = lambda opt: (N, )
    
    # Launch kernel
    softmax_kernel[grid](input_tensor, output_tensor, input_tensor.stride(0), output_tensor.stride(0), N, M, BLOCK_SIZE)
    
    return output_tensor.to(input_tensor.dtype)

# Example usage
if __name__ == "__main__":
    input_tensor = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
    output_tensor = kernel(input_tensor)
    print(output_tensor)
