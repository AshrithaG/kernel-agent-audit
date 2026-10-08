"""An LLM writes Triton kernels over several rounds, steered by one harness's feedback.

For every task, two conditions run side by side from the same first prompt:

  naive_feedback      after each round the model is told what the naive harness says
  rigorous_feedback   ... what the rigorous harness says

Each round asks for K candidates; every candidate is judged by BOTH harnesses and
recorded, but only the condition's own harness picks the best one and writes the
feedback for the next round. So the record shows, for every kernel, what a quick
pipeline would have claimed and what holds up, and whether steering by the quick
harness drifts the model toward kernels that please it.

The model is any OpenAI-compatible endpoint (vLLM's server on the GPU machine).

    python -m kaa.agent --base-url http://localhost:8001 --model Qwen/Qwen2.5-Coder-14B-Instruct-AWQ \\
        --rounds 3 --samples 4 --out results/run1
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from kaa.tasks import TASKS

SYSTEM = ("You are an expert GPU kernel engineer. You write correct, fast Triton kernels "
          "for NVIDIA GPUs. Reply with one complete Python file in a single ```python block.")

CONTRACT = """Write a Python file that defines `kernel(*inputs)` returning exactly what `reference(*inputs)` returns
(same shape, same dtype, same values within floating-point tolerance), implemented with at least one
Triton kernel (`import triton`, `import triton.language as tl`, a function decorated with `@triton.jit`).
It must work for any input shape of the form described, not just one size. You may import only torch,
triton and the Python standard math library. The inputs are float16 CUDA tensors (integer targets are int64)."""


# What an agent setup would give a model about Triton: the language's common traps and one
# complete kernel for an op that is not among the tasks (scale each row by a vector).
TRITON_NOTES = """Triton notes (Triton 3.7):
- A @triton.jit kernel runs once per program instance; there are no thread indices. Use
  tl.program_id(axis) for the block and tl.arange(0, BLOCK) for the elements in it.
- BLOCK sizes must be tl.constexpr powers of two; mask every load and store with a bound check.
- Use x * x, not x ** 2. Use tl.exp, tl.log, tl.sqrt, tl.rsqrt, tl.sigmoid, tl.maximum; tanh and
  erf come from triton.language.extra.libdevice. Python's math functions only work on constants.
- Reductions: tl.sum, tl.max, tl.min over an axis of a block. Shapes inside a kernel must be
  compile-time constants.
- Do the arithmetic in float32 (x.to(tl.float32)) and cast the result back when storing.
- Launch from Python: kernel[grid](args..., BLOCK=...), where grid is a tuple of ints.

Example, a complete file for a different op (multiply each row of x by a vector s):
```python
import torch
import triton
import triton.language as tl

@triton.jit
def _scale_rows(x_ptr, s_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * n_cols + cols, (x * s).to(out_ptr.dtype.element_ty), mask=mask)

def kernel(x, s):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    _scale_rows[(n_rows,)](x, s, out, n_cols, BLOCK=triton.next_power_of_2(n_cols))
    return out
```"""


def first_prompt(task) -> str:
    return (f"Task: {task.description}\n\nReference implementation:\n```python\n{task.source}```\n\n"
            f"{CONTRACT}\n\n{TRITON_NOTES}\n\nGoal: make it as fast as possible on an RTX 4090.")


def feedback(cond: str, res: dict) -> str:
    if "static" in res:
        return f"Your file was rejected before running: {res['static']}."
    if "load_error" in res:
        return f"Your file failed to import: {res['load_error']}."
    if res.get("timeout"):
        return "Your kernel timed out."
    if cond == "naive_feedback":
        n = res["naive"]
        if not n["correct"]:
            return f"Your kernel is incorrect ({n['reason']}: {n.get('detail', '')})."
        return f"Your kernel is correct and runs at {n['speedup']:.2f}x the speed of the PyTorch reference."
    r = res["rigorous"]
    if not r["correct"]:
        return (f"Your kernel failed the check '{r['reason']}' on input shape {r.get('shape')}: "
                f"{r.get('detail', '')}.")
    lines = [f"  shape {s['shape']}: {s.get('vs_eager', float('nan')):.2f}x eager PyTorch, "
             f"{s.get('vs_compile', float('nan')):.2f}x torch.compile" for s in r["per_shape"]]
    return ("Your kernel is correct on every shape. Speed, CUDA-event timing with a cold L2 cache:\n"
            + "\n".join(lines))


def score(cond: str, res: dict) -> float:
    """How the condition's own harness ranks a candidate: correct ones by speed, others last."""
    if "naive" not in res:
        return -2.0
    if cond == "naive_feedback":
        return res["naive"]["speedup"] if res["naive"]["correct"] else -1.0
    r = res["rigorous"]
    return (r.get("speedup_vs_eager") or 0.0) if r["correct"] else -1.0


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    return max(blocks, key=len) if blocks else text


def chat(base_url: str, model: str, messages: list, n: int, temperature: float, max_tokens: int) -> list[str]:
    body = json.dumps({"model": model, "messages": messages, "n": n, "temperature": temperature,
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return [c["message"]["content"] for c in json.loads(r.read())["choices"]]


def run_candidate(task: str, code_path: Path, baselines: Path, timeout: int) -> dict:
    # Absolute paths: the evaluator runs with the code's folder as its working directory,
    # so a relative --code path would point nowhere (the first GPU run lost every
    # candidate to exactly that).
    code_path, baselines = code_path.resolve(), baselines.resolve()
    env = {k: os.environ[k] for k in ("PATH", "HOME", "CUDA_VISIBLE_DEVICES", "PYTHONPATH",
                                      "TRITON_CACHE_DIR", "LD_LIBRARY_PATH") if k in os.environ}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        p = subprocess.run([sys.executable, "-m", "kaa.evaluate", "--task", task, "--code", str(code_path),
                            "--baselines", str(baselines)], capture_output=True, text=True,
                           timeout=timeout, env=env, cwd=code_path.parent)
    except subprocess.TimeoutExpired:
        return {"timeout": True}
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        return {"load_error": f"evaluator exited {p.returncode}: {(p.stderr or '').strip()[-300:]}"}
    return json.loads(lines[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8001")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", nargs="*", default=sorted(TASKS))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--baselines", type=Path, default=Path("results/baselines.json"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "code").mkdir(exist_ok=True)
    log = (a.out / "candidates.jsonl").open("a")
    conds = ["naive_feedback", "rigorous_feedback"]
    state = {(t, c): {"messages": [{"role": "system", "content": SYSTEM},
                                   {"role": "user", "content": first_prompt(TASKS[t])}],
                      "best": None} for t in a.tasks for c in conds}
    (a.out / "config.json").write_text(json.dumps(vars(a), default=str, indent=1))

    for rnd in range(a.rounds):
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=16) as ex:
            futs = {k: ex.submit(chat, a.base_url, a.model, s["messages"], a.samples, a.temperature,
                                 a.max_tokens) for k, s in state.items()}
            replies = {k: f.result() for k, f in futs.items()}
        print(f"round {rnd}: generated in {time.time() - t0:.0f}s", flush=True)
        for (t, c), texts in replies.items():
            s = state[(t, c)]
            results = []
            for i, text in enumerate(texts):
                code = extract_code(text)
                h = hashlib.sha1(code.encode()).hexdigest()[:12]
                path = a.out / "code" / f"{t}_{c}_r{rnd}_s{i}_{h}.py"
                path.write_text(code)
                res = run_candidate(t, path, a.baselines.resolve(), a.timeout)
                rec = {"task": t, "cond": c, "round": rnd, "sample": i, "hash": h, "path": str(path),
                       "score": score(c, res), **res}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                results.append((score(c, res), text, res, rec))
                n, r = res.get("naive", {}), res.get("rigorous", {})
                print(f"  {t:17s} {c:17s} r{rnd} s{i}: naive {n.get('correct')} "
                      f"{n.get('speedup', 0):.2f}x | rigorous {r.get('correct')} {r.get('reason', '')} "
                      f"{(r.get('speedup_vs_eager') or 0):.2f}x eager", flush=True)
            best = max(results, key=lambda x: x[0])
            if s["best"] is None or best[0] >= s["best"][0]:
                s["best"] = best
            # Next round: the conversation continues from the best answer so far and its feedback.
            _, text, res, rec = s["best"]
            s["messages"] = s["messages"][:2] + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": feedback(c, res) + "\n\nImprove the kernel. Reply with the "
                 "complete file in one ```python block."}]
            s["final"] = rec
    (a.out / "final.json").write_text(json.dumps(
        {f"{t}|{c}": state[(t, c)]["final"] for (t, c) in state}, indent=1))


if __name__ == "__main__":
    main()
