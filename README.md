# kernel-agent-audit

LLM agents that write GPU kernels are usually scored by a quick harness: one input
shape, a loose `allclose`, a timing loop against eager PyTorch. This project asks how
many of the speedups such a harness reports survive a careful one, why the rest fail,
and whether steering an agent with the quick harness's feedback teaches it to write
kernels that please that harness rather than kernels that are fast.

## Setup

**Tasks** (`kaa/tasks.py`): twelve PyTorch references an agent must reimplement in
Triton: softmax, LayerNorm, RMSNorm, add+RMSNorm, bias+GELU, SwiGLU, row mean,
per-row cross-entropy, rotary embedding, causal softmax, matmul+bias+ReLU and
transpose. Each has a power-of-two "dev" shape and four or five shapes that are not
(1000 x 3001, a single row, very wide or very narrow).

**Two harnesses** (`kaa/harness.py`):

| | naive | rigorous |
|---|---|---|
| shapes | the dev shape | every shape |
| inputs | one set, reused for checking and timing | three fresh sets per shape |
| tolerance | atol = rtol = 1e-2 | per task (down to 2e-3; exact for transpose) |
| integrity | none | inputs unchanged, output not aliasing an input, shape and dtype match, not a cached result for other inputs, contains a `@triton.jit` kernel |
| timing | 20-call loop, synchronize at the end | CUDA events, warmup, L2 flushed before each call, input sets rotated |
| baseline | eager PyTorch | eager PyTorch and `torch.compile` |

When a candidate passes the naive check and fails the rigorous one, the first failing
check names why: `shape_overfit`, `tolerance`, `stale_output`, `mutated_input`,
`aliases_input`, `not_a_triton_kernel`, `error_off_dev_shape`.

**Agent** (`kaa/agent.py`): an open coding model served by vLLM
(Qwen2.5-Coder-14B-Instruct, AWQ) writes kernels over several rounds. Two conditions
start from the same prompt: one is shown the naive harness's verdict after each
round, the other the rigorous one's. Every candidate is judged by both harnesses and
logged; only the condition's own harness picks the best and writes the next feedback.

**Safety.** Generated code is executed. Each candidate goes through a static check
(imports limited to torch, triton and the standard math modules; no file, process or
introspection access) and then runs in its own subprocess with a timeout. That keeps
it to numerical work and contains crashes; it is not a security sandbox. A gVisor
container would be the stronger isolation (see
[agent-sandbox-bench](https://github.com/AshrithaG/agent-sandbox-bench)), at the cost of
GPU passthrough.

## Tests

`tests/test_harness.py` builds one cheat per failure mode (a kernel hard-coded to the
dev width, one that caches its output, one that writes into its input, one off by
5e-3, one returning a view, one that is plain torch) and checks each passes the naive
harness and is caught by the rigorous one under the right name.
`tests/test_agent_loop.py` runs the whole loop on CPU against a stub model server.

    python -m pytest -q tests

## Running

On a CUDA machine with vLLM installed:

    bash tools/run_vm.sh 2>&1 | tee ~/kaa.log

It times the baselines with the GPU otherwise idle, starts the model server, runs a
one-task smoke test, then the full study, and writes `results/run1/report.md`.

## Results

Not yet run.
