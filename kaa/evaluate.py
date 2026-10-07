"""Judge one candidate with both harnesses, in a process of its own.

    python -m kaa.evaluate --task softmax --code cand.py --baselines results/baselines.json

Prints one JSON line. The agent runs this in a subprocess with a timeout, so a
candidate that hangs, crashes the CUDA context or leaks memory takes nothing else
with it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kaa.harness import load, naive_eval, rigorous_eval, static_check, _err
from kaa.tasks import TASKS


def evaluate(task_name: str, path: Path, baselines: dict | None, timing: bool = True) -> dict:
    task = TASKS[task_name]
    code = Path(path).read_text()
    bad = static_check(code)
    if bad:
        return {"static": bad}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        fn = load(path)
    except Exception as e:
        return {"load_error": _err(e)}
    return {"naive": naive_eval(task, fn, device, dtype),
            "rigorous": rigorous_eval(task, fn, device, dtype, baselines, code=code, timing=timing)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--code", type=Path, required=True)
    ap.add_argument("--baselines", type=Path)
    ap.add_argument("--no-timing", action="store_true")
    a = ap.parse_args()
    base = json.loads(a.baselines.read_text()) if a.baselines and a.baselines.exists() else None
    print(json.dumps(evaluate(a.task, a.code, base, timing=not a.no_timing)))


if __name__ == "__main__":
    main()
