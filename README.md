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
one-task smoke test (which stops everything if no candidate runs), then the full study,
and writes `results/run3/report.md`.

## Results

One run on an RTX 4090 (`results/run3/`): Qwen2.5-Coder-14B-Instruct (AWQ, served by vLLM),
12 tasks, both feedback conditions, 4 rounds of 4 candidates, 384 candidates in all.
`results/run3/report.md` has every table; the numbers below are from it.

**Claimed speedups over eager PyTorch mostly hold up. None beat torch.compile.**

| | candidates |
|---|---|
| ran | 378 |
| passed the naive check | 95 (25%) |
| ...and the rigorous check | 87 |
| naive harness says faster than PyTorch | 75 |
| ...still faster than eager PyTorch under the rigorous harness | **70 of 75** |
| ...faster than torch.compile under the rigorous harness | **0 of 75** |

The 8 candidates that passed the naive check and failed the rigorous one were all
`shape_overfit`: right at the dev shape, wrong on others. Six of them cap the block at 1,024
or 2,048 columns while each program handles one whole row, several with a comment like
"limit block size to 1024 to avoid too many threads", a CUDA thread-limit habit that does
not apply to Triton. Rows wider than the cap silently lose their remaining columns, and the
dev shape is exactly 1,024 wide, so the naive check cannot see it. No candidate cached its
output, wrote into its input or returned a view.

Careful timing did not deflate the speedups. Among the 87 kernels correct under both
harnesses the naive speedup has a median of 2.98x and the rigorous one 4.36x, and the two
rank kernels alike (Kendall tau 0.81): flushing the L2 cache costs the eager reference,
several kernels with intermediates in memory, more than a single fused kernel. What
flatters these kernels is the baseline. The best correct kernel per task:

| task | vs eager PyTorch | vs torch.compile |
|---|---|---|
| rope | 6.55x | 0.81x |
| rmsnorm | 4.82x | 0.96x |
| causal_softmax | 4.43x | 0.95x |
| layernorm | 3.41x | 0.89x |
| row_mean | 2.13x | 0.93x |
| bias_gelu | 1.66x | 0.45x |
| softmax | 1.45x | 0.54x |

No correct kernel came out for add_rmsnorm, cross_entropy, matmul_bias_relu, swiglu or
transpose.

**Steering by the naive harness did not teach the model to game it.** Over all rounds the
rigorous-feedback condition produced more correct kernels (50 against 37 of 189 each) and
more tasks with a final kernel faster than eager (6 against 4 of 12), and it also had more
of the shape-overfit failures (7 against 1). With one model and one seed, 50 against 37 is
not significant (p about 0.11). Neither condition improved much over rounds.

**What decided success was knowing Triton.** The first round of an earlier run used the same
model and tasks with no Triton notes in the prompt: 0 of 96 first attempts were correct.
With one page of notes (the language's common traps) and one worked kernel for an op that is
not among the tasks, 28 of 96 first attempts passed the naive check and 26 the rigorous one
(z = 5.5). Without the notes the model wrote CUDA habits into Triton: `tl.thread_idx`,
`x ** 2` on tensors, `tl.tanh`, Python `math` calls inside kernels.

## How the runs went

1. **Run 1** lost every candidate: the evaluator ran with the code's folder as its working
   directory and was handed a relative path. Not kept.
2. **A smoke run** then stopped the next attempt before the study: candidates were executed
   from a string, and Triton's `@jit` cannot read source from one.
3. **Run 2** (`results/run2/`) ran in full, but every error fed back to the model was only its
   first line, which for a Triton compile error is a position with no cause, and 20
   candidates were rejected for importing `time` or `numpy` or naming a parameter `input`.
   Its first round is the no-notes comparison above. The file also holds an attempt
   interrupted by hand; the analysis uses the complete run that follows it.
4. **Run 3** (`results/run3/`) is the study: full error causes fed back, harmless imports
   allowed, Triton notes in the prompt, 4 rounds.

Each fix came with a test that fails without it.

## Limits

One model, one seed, one GPU, 12 tasks. A stronger model would produce more correct
kernels; whether its claims would also hold against eager and fail against torch.compile is
the obvious next run.
