import torch
import triton
import triton.language as tl
import math

# Triton kernel to compute the mean over the last dimension of a 2-D tensor
@triton.jit
def mean_kernel(
    X,  # input matrix
    Y,  # output vector
    N,  # number of rows
    M,  # number of columns
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    cols_per_pid = tl.cdiv(M, BLOCK_M)
    col_offset = pid * BLOCK_M
    mask = tl.arange(0, BLOCK_M) < M - col_offset
    X = X[pid, col_offset : col_offset + BLOCK_M]
    acc = tl.sum(X[mask], axis=0)
    total_elements = tl.sum(mask, axis=0)
    Y[pid] = acc.to(tl.float32) / total_elements

# Function to launch the Triton kernel
def kernel(X):
    N, M = X.shape
    Y = torch.empty(N, dtype=torch.float32, device=X.device)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_M']),)
    BLOCK_M = 128
    mean_kernel[grid](X, Y, N, M, BLOCK_M=BLOCK_M)
    return Y.to(X.dtype)

# Reference implementation for comparison
def reference(X):
    return X.float().mean(dim=-1).to(X.dtype)

# Example usage
if __name__ == "__main__":
    X = torch.rand((1024, 512), dtype=torch.float16, device='cuda')
    Y_triton = kernel(X)
    Y_ref = reference(X)
    print("Triton output:", Y_triton)
    print("Reference output:", Y_ref)
    assert torch.allclose(Y_triton, Y_ref, atol=1e-3, rtol=1e-3), "The outputs do not match"
