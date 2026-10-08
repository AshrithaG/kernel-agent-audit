# Kernel agent audit: run3

384 candidates.
378 ran; 1 rejected by the static check, 4 failed to import, 1 timed out.

## Correctness

Passed the naive check: 95/378 (25%). Of those, passed the rigorous check: 87/95 (92%).

Why naive-passing candidates failed the rigorous check:

| reason | candidates |
|---|---|
| shape_overfit | 8 |

## Speedup claims

Candidates the naive harness calls faster than PyTorch: 75.
Still faster than eager PyTorch under the rigorous harness: 70/75 (93%).
Faster than torch.compile under the rigorous harness: 0/75 (0%).
Among candidates correct under both (87), the naive speedup's median is 2.98x and the rigorous one's 4.36x; Kendall tau between them 0.81.

## Final kernels per task and condition

| task | condition | naive says | rigorous: correct / reason | vs eager | vs torch.compile |
|---|---|---|---|---|---|
| add_rmsnorm | naive_feedback | error | error | - | - |
| add_rmsnorm | rigorous_feedback | wrong | wrong | - | - |
| bias_gelu | naive_feedback | 1.69x | shape_overfit | - | - |
| bias_gelu | rigorous_feedback | 1.63x | yes | 1.62x | 0.44x |
| causal_softmax | naive_feedback | 3.18x | yes | 4.36x | 0.94x |
| causal_softmax | rigorous_feedback | error | error | - | - |
| cross_entropy | naive_feedback | error | error | - | - |
| cross_entropy | rigorous_feedback | error | error | - | - |
| layernorm | naive_feedback | error | error | - | - |
| layernorm | rigorous_feedback | 1.80x | yes | 3.41x | 0.89x |
| matmul_bias_relu | naive_feedback | error | error | - | - |
| matmul_bias_relu | rigorous_feedback | error | error | - | - |
| rmsnorm | naive_feedback | 3.39x | yes | 4.81x | 0.95x |
| rmsnorm | rigorous_feedback | 3.28x | yes | 4.82x | 0.96x |
| rope | naive_feedback | 5.14x | yes | 6.32x | 0.78x |
| rope | rigorous_feedback | 4.97x | yes | 6.23x | 0.77x |
| row_mean | naive_feedback | error | error | - | - |
| row_mean | rigorous_feedback | 0.92x | yes | 2.13x | 0.93x |
| softmax | naive_feedback | 1.01x | yes | 1.45x | 0.54x |
| softmax | rigorous_feedback | 0.97x | yes | 1.45x | 0.54x |
| swiglu | naive_feedback | error | error | - | - |
| swiglu | rigorous_feedback | error | error | - | - |
| transpose | naive_feedback | error | error | - | - |
| transpose | rigorous_feedback | error | error | - | - |

## Does the feedback harness change what the model writes?

| condition | round | candidates | naive-correct | rigorous-correct | claimed | real (vs eager) |
|---|---|---|---|---|---|---|
| naive_feedback | 0 | 46 | 12 | 12 | 9 | 12 |
| naive_feedback | 1 | 48 | 9 | 9 | 8 | 9 |
| naive_feedback | 2 | 48 | 8 | 8 | 6 | 8 |
| naive_feedback | 3 | 47 | 9 | 8 | 8 | 8 |
| rigorous_feedback | 0 | 47 | 16 | 14 | 11 | 14 |
| rigorous_feedback | 1 | 46 | 13 | 12 | 11 | 12 |
| rigorous_feedback | 2 | 48 | 12 | 11 | 10 | 11 |
| rigorous_feedback | 3 | 48 | 16 | 13 | 12 | 13 |

Final kernels, naive_feedback: real wins over eager 4/12 (33%), over torch.compile 0/12 (0%).

Final kernels, rigorous_feedback: real wins over eager 6/12 (50%), over torch.compile 0/12 (0%).
