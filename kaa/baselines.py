"""Eager and torch.compile timings of every task's reference, at every shape, with the
rigorous harness's timing method. Run once, with nothing else on the GPU.

    python -m kaa.baselines --out results/baselines.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kaa.harness import input_sets, time_us
from kaa.tasks import TASKS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/baselines.json"))
    ap.add_argument("--tasks", nargs="*", default=sorted(TASKS))
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    res = {"device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
           "torch": torch.__version__}
    for name in a.tasks:
        task = TASKS[name]
        compiled = torch.compile(task.reference, dynamic=False)
        res[name] = {}
        for shape in task.shapes:
            sets = input_sets(task, shape, device, dtype)
            row = {"eager_us": time_us(task.reference, sets, device)}
            try:
                row["compile_us"] = time_us(compiled, sets, device)
            except Exception as e:  # recorded, not fatal: some baselines may not compile
                row["compile_error"] = str(e).splitlines()[0][:200]
            res[name]["x".join(map(str, shape))] = row
            print(name, shape, row, flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
