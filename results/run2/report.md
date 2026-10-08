# Kernel agent audit: run2

459 candidates.
418 ran; 41 rejected by the static check, 0 failed to import, 0 timed out.

## Correctness

Passed the naive check: 5/418 (1%). Of those, passed the rigorous check: 1/5 (20%).

Why naive-passing candidates failed the rigorous check:

| reason | candidates |
|---|---|
| shape_overfit | 4 |

## Speedup claims

Candidates the naive harness calls faster than PyTorch: 4.
Still faster than eager PyTorch under the rigorous harness: 1/4 (25%).
Faster than torch.compile under the rigorous harness: 0/4 (0%).


## Final kernels per task and condition

| task | condition | naive says | rigorous: correct / reason | vs eager | vs torch.compile |
|---|---|---|---|---|---|
| add_rmsnorm | naive_feedback | error | error | - | - |
| add_rmsnorm | rigorous_feedback | error | error | - | - |
| bias_gelu | naive_feedback | error | error | - | - |
| bias_gelu | rigorous_feedback | error | error | - | - |
| causal_softmax | naive_feedback | error | error | - | - |
| causal_softmax | rigorous_feedback | error | error | - | - |
| cross_entropy | naive_feedback | error | error | - | - |
| cross_entropy | rigorous_feedback | error | error | - | - |
| layernorm | naive_feedback | error | error | - | - |
| layernorm | rigorous_feedback | error | error | - | - |
| matmul_bias_relu | naive_feedback | 1.39x | shape_overfit | - | - |
| matmul_bias_relu | rigorous_feedback | error | error | - | - |
| rmsnorm | naive_feedback | error | error | - | - |
| rmsnorm | rigorous_feedback | error | error | - | - |
| rope | naive_feedback | wrong | wrong | - | - |
| rope | rigorous_feedback | error | error | - | - |
| row_mean | naive_feedback | error | error | - | - |
| row_mean | rigorous_feedback | error | error | - | - |
| softmax | naive_feedback | error | error | - | - |
| softmax | rigorous_feedback | wrong | wrong | - | - |
| swiglu | naive_feedback | error | error | - | - |
| swiglu | rigorous_feedback | error | error | - | - |
| transpose | naive_feedback | wrong | wrong | - | - |
| transpose | rigorous_feedback | error | error | - | - |

## Does the feedback harness change what the model writes?

| condition | round | candidates | naive-correct | rigorous-correct | claimed | real (vs eager) |
|---|---|---|---|---|---|---|
| naive_feedback | 0 | 74 | 0 | 0 | 0 | 0 |
| naive_feedback | 1 | 87 | 2 | 1 | 2 | 1 |
| naive_feedback | 2 | 48 | 3 | 0 | 2 | 0 |
| rigorous_feedback | 0 | 78 | 0 | 0 | 0 | 0 |
| rigorous_feedback | 1 | 83 | 0 | 0 | 0 | 0 |
| rigorous_feedback | 2 | 48 | 0 | 0 | 0 | 0 |

Final kernels, naive_feedback: real wins over eager 0/12 (0%), over torch.compile 0/12 (0%).

Final kernels, rigorous_feedback: real wins over eager 0/12 (0%), over torch.compile 0/12 (0%).
