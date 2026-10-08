"""Two ways to judge a generated kernel: the quick way and the careful way.

naive_eval is what a quick agent pipeline tends to do: one shape (the task's
power-of-two dev shape), one input set reused for checking and timing, a loose
allclose, and a timing loop with a synchronize at the end, against eager PyTorch.

rigorous_eval checks what the naive one cannot see:
  * every shape in the task, three fresh input sets each, at the task's tolerance;
  * the candidate is a Triton kernel at all (not a call back into torch);
  * inputs are unchanged after the call, the output does not alias an input,
    and its shape and dtype match the reference's;
  * the output is not a stale result from an earlier call (caching);
and times every shape with CUDA events, a warmup, the L2 cache flushed before
every timed call and input sets rotated, against both eager PyTorch and
torch.compile of the reference.

When a candidate passes the naive check and fails the rigorous one, the first
failing check names the reason. Those names are the categories in the report.
"""

from __future__ import annotations

import ast
import math
import statistics
import time

import torch

NAIVE_ATOL = NAIVE_RTOL = 1e-2
# time and numpy are harmless and models reach for them; "input" was dropped from the
# forbidden names because models name kernel parameters `input` (the first study run
# rejected 20 candidates for these reasons alone).
ALLOWED_IMPORTS = {"torch", "triton", "math", "functools", "typing", "itertools", "operator",
                   "dataclasses", "__future__", "time", "numpy"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "breakpoint",
                   "globals", "locals", "vars"}
FORBIDDEN_ATTRS = {"system", "popen", "spawn", "fork", "remove", "unlink", "rmtree",
                   "__subclasses__", "__globals__", "__builtins__", "__code__"}


# ---------------------------------------------------------------- loading
def static_check(code: str) -> str | None:
    """Reject code that reaches outside numerical work. Not a security sandbox (the
    runner adds a subprocess and a timeout); it keeps generated code to its job."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                if m.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"import of {m} not allowed"
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return f"use of {node.id} not allowed"
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            return f"use of .{node.attr} not allowed"
    if not any(isinstance(n, ast.FunctionDef) and n.name == "kernel" for n in tree.body):
        return "no top-level function named kernel"
    return None


def has_triton_kernel(code: str) -> bool:
    """At least one function decorated with triton.jit (or triton.autotune over one)."""
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.FunctionDef):
            for d in node.decorator_list:
                text = ast.unparse(d)
                if "triton.jit" in text or text in ("jit", "triton.autotune") or "autotune" in text:
                    return True
    return False


def load(path) -> object:
    """Import the candidate from its file. Triton's @jit reads a function's source with
    inspect, which only works for code that lives in a real file: executing the code
    from a string fails with "@jit functions should be defined in a Python file" (the
    second GPU run lost every candidate to that)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"candidate_{abs(hash(str(path)))}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # after static_check
    return mod.kernel


# ---------------------------------------------------------------- helpers
def _gen(device: str, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _err(e: BaseException) -> str:
    """The error's type, first line and last lines. A Triton CompilationError's first line
    is only a position ("at 27:25:"); the cause is at the end, after the code excerpt, and
    the model needs it to fix anything (the first study run fed back positions only)."""
    lines = [ln for ln in str(e).strip().splitlines() if ln.strip()]
    if not lines:
        return type(e).__name__
    text = lines[0] if len(lines) == 1 else f"{lines[0]} ... {' '.join(lines[-2:])}"
    return f"{type(e).__name__}: {text}"[:600]


def _close(out, ref, atol, rtol) -> bool:
    return bool(torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol, equal_nan=True))


def _max_err(out, ref) -> float:
    d = (out.float() - ref.float()).abs()
    d = d[torch.isfinite(d)]
    return float(d.max()) if d.numel() else float("nan")


def _aliases(out, inputs) -> bool:
    p = out.untyped_storage().data_ptr()
    return any(isinstance(x, torch.Tensor) and x.untyped_storage().data_ptr() == p for x in inputs)


# ---------------------------------------------------------------- naive
def naive_eval(task, fn, device: str, dtype: torch.dtype, iters: int = 20) -> dict:
    inputs = task.make_inputs(task.dev_shape, _gen(device, 0), device, dtype)
    ref = task.reference(*inputs)
    try:
        out = fn(*inputs)
        _sync(device)
    except Exception as e:
        return {"correct": False, "reason": "error", "detail": _err(e)}
    if not isinstance(out, torch.Tensor) or out.shape != ref.shape or not _close(out, ref, NAIVE_ATOL, NAIVE_RTOL):
        return {"correct": False, "reason": "wrong", "detail": "allclose failed on the dev shape"}

    def loop(f) -> float:
        f()
        _sync(device)
        t = time.perf_counter()
        for _ in range(iters):
            f()
        _sync(device)
        return (time.perf_counter() - t) / iters * 1e6

    t_ref = loop(lambda: task.reference(*inputs))
    t = loop(lambda: fn(*inputs))
    return {"correct": True, "speedup": t_ref / t, "t_ref_us": t_ref, "t_us": t}


# ---------------------------------------------------------------- rigorous
def _fail(reason: str, shape, detail: str) -> dict:
    return {"correct": False, "reason": reason, "shape": list(shape), "detail": detail}


def check(task, fn, device: str, dtype: torch.dtype, code: str | None = None) -> dict:
    """Correctness and integrity on every shape; returns {"correct": True} or the first failure."""
    if code is not None and not has_triton_kernel(code):
        return _fail("not_a_triton_kernel", task.dev_shape, "no @triton.jit function in the candidate")
    for shape in task.shapes:
        for seed in (1, 2, 3):
            inputs = task.make_inputs(shape, _gen(device, seed), device, dtype)
            clones = [x.clone() for x in inputs]
            ref = task.reference(*clones)
            try:
                out = fn(*inputs)
                _sync(device)
            except Exception as e:
                return _fail("error_off_dev_shape" if tuple(shape) != tuple(task.dev_shape) else "error",
                             shape, _err(e))
            if not isinstance(out, torch.Tensor):
                return _fail("bad_output", shape, f"returned {type(out).__name__}")
            if out.shape != ref.shape or out.dtype != ref.dtype:
                return _fail("shape_or_dtype", shape, f"got {tuple(out.shape)} {out.dtype}, "
                             f"want {tuple(ref.shape)} {ref.dtype}")
            if any(not torch.equal(x, c) for x, c in zip(inputs, clones)):
                return _fail("mutated_input", shape, "an input changed during the call")
            if _aliases(out, inputs):
                return _fail("aliases_input", shape, "the output shares storage with an input")
            if not _close(out, ref, task.atol, task.rtol):
                if _stale(task, out, shape, seed, device, dtype):
                    return _fail("stale_output", shape, "returned the result for a different input "
                                 "of the same shape (a cached output)")
                loose = _close(out, ref, NAIVE_ATOL, NAIVE_RTOL)
                if tuple(shape) == tuple(task.dev_shape):
                    reason = "tolerance" if loose else "wrong"
                else:
                    reason = "tolerance" if loose else "shape_overfit"
                return _fail(reason, shape, f"max abs error {_max_err(out, ref):.3g} "
                             f"(atol {task.atol}, rtol {task.rtol})")
    return {"correct": True}


def _stale(task, out, shape, seed, device, dtype) -> bool:
    """Does `out` match the reference for another input set of this shape (seed 0 is the
    naive harness's), i.e. did the candidate return a result it computed earlier?"""
    for other in (0, 1, 2, 3):
        if other != seed:
            ref = task.reference(*task.make_inputs(shape, _gen(device, other), device, dtype))
            if out.shape == ref.shape and _close(out, ref, task.atol, task.rtol):
                return True
    return False


def time_us(fn, input_sets, device: str, warmup: int = 10, iters: int = 30) -> float:
    """Median microseconds per call: CUDA events, L2 flushed before each call, inputs rotated."""
    for i in range(warmup):
        fn(*input_sets[i % len(input_sets)])
    _sync(device)
    times = []
    if device == "cuda":
        flush = torch.empty(128 * 2**20, dtype=torch.uint8, device=device)  # > the 4090's 72 MB L2
        for i in range(iters):
            flush.zero_()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn(*input_sets[i % len(input_sets)])
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b) * 1e3)
    else:
        for i in range(iters):
            t = time.perf_counter()
            fn(*input_sets[i % len(input_sets)])
            times.append((time.perf_counter() - t) * 1e6)
    return statistics.median(times)


def input_sets(task, shape, device, dtype, n: int = 3):
    return [task.make_inputs(shape, _gen(device, 100 + i), device, dtype) for i in range(n)]


def rigorous_eval(task, fn, device: str, dtype: torch.dtype, baselines: dict | None,
                  code: str | None = None, timing: bool = True) -> dict:
    res = check(task, fn, device, dtype, code)
    if not res["correct"] or not timing:
        return res
    per_shape = []
    for shape in task.shapes:
        key = "x".join(map(str, shape))
        t = time_us(fn, input_sets(task, shape, device, dtype), device)
        base = (baselines or {}).get(task.name, {}).get(key, {})
        row = {"shape": list(shape), "t_us": t}
        if base.get("eager_us"):
            row["vs_eager"] = base["eager_us"] / t
        if base.get("compile_us"):
            row["vs_compile"] = base["compile_us"] / t
        per_shape.append(row)

    def geomean(k):
        v = [r[k] for r in per_shape if k in r]
        return math.exp(statistics.mean(math.log(x) for x in v)) if v else None

    res.update(per_shape=per_shape, speedup_vs_eager=geomean("vs_eager"),
               speedup_vs_compile=geomean("vs_compile"))
    return res
