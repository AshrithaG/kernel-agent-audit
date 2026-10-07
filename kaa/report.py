"""Summarize an agent run: how many claimed speedups survive, why the rest don't, and
whether steering by the naive harness changed what the model produced.

    python -m kaa.report results/run1
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def load(run: Path) -> list[dict]:
    return [json.loads(ln) for ln in (run / "candidates.jsonl").read_text().splitlines() if ln.strip()]


def claimed(r: dict) -> bool:
    n = r.get("naive", {})
    return bool(n.get("correct")) and n.get("speedup", 0) > 1.0


def real(r: dict, vs: str = "speedup_vs_eager") -> bool:
    g = r.get("rigorous", {})
    return bool(g.get("correct")) and (g.get(vs) or 0) > 1.0


def kendall_tau(xs, ys) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    sign = lambda v: (v > 0) - (v < 0)
    s = sum(sign(xs[i] - xs[j]) * sign(ys[i] - ys[j]) for i in range(n) for j in range(i + 1, n))
    return s / (n * (n - 1) / 2)


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "0/0"


def main(run: Path) -> str:
    recs = load(run)
    out = [f"# Kernel agent audit: {run.name}", "", f"{len(recs)} candidates."]
    ran = [r for r in recs if "naive" in r]
    out.append(f"{len(ran)} ran; {sum('static' in r for r in recs)} rejected by the static check, "
               f"{sum('load_error' in r for r in recs)} failed to import, "
               f"{sum(bool(r.get('timeout')) for r in recs)} timed out.")

    passed_naive = [r for r in ran if r["naive"]["correct"]]
    passed_both = [r for r in passed_naive if r["rigorous"]["correct"]]
    out += ["", "## Correctness", "",
            f"Passed the naive check: {pct(len(passed_naive), len(ran))}. "
            f"Of those, passed the rigorous check: {pct(len(passed_both), len(passed_naive))}.", "",
            "Why naive-passing candidates failed the rigorous check:", "",
            "| reason | candidates |", "|---|---|"]
    for reason, n in Counter(r["rigorous"]["reason"] for r in passed_naive
                             if not r["rigorous"]["correct"]).most_common():
        out.append(f"| {reason} | {n} |")

    claims = [r for r in ran if claimed(r)]
    out += ["", "## Speedup claims", "",
            f"Candidates the naive harness calls faster than PyTorch: {len(claims)}.",
            f"Still faster than eager PyTorch under the rigorous harness: "
            f"{pct(sum(real(r) for r in claims), len(claims))}.",
            f"Faster than torch.compile under the rigorous harness: "
            f"{pct(sum(real(r, 'speedup_vs_compile') for r in claims), len(claims))}."]
    both = [r for r in passed_both if r["rigorous"].get("speedup_vs_eager")]
    if both:
        xs = [r["naive"]["speedup"] for r in both]
        ys = [r["rigorous"]["speedup_vs_eager"] for r in both]
        tau = kendall_tau(xs, ys)
        out.append(f"Among candidates correct under both ({len(both)}), the naive speedup's median is "
                   f"{statistics.median(xs):.2f}x and the rigorous one's {statistics.median(ys):.2f}x; "
                   f"Kendall tau between them {tau:.2f}." if tau is not None else "")

    out += ["", "## Final kernels per task and condition", "",
            "| task | condition | naive says | rigorous: correct / reason | vs eager | vs torch.compile |",
            "|---|---|---|---|---|---|"]
    final = json.loads((run / "final.json").read_text()) if (run / "final.json").exists() else {}
    for key in sorted(final):
        r = final[key]
        t, c = key.split("|")
        n, g = r.get("naive", {}), r.get("rigorous", {})
        ns = f"{n['speedup']:.2f}x" if n.get("correct") else (n.get("reason") or "not run")
        gs = "yes" if g.get("correct") else (g.get("reason") or "not run")
        ve = f"{g['speedup_vs_eager']:.2f}x" if g.get("speedup_vs_eager") else "-"
        vc = f"{g['speedup_vs_compile']:.2f}x" if g.get("speedup_vs_compile") else "-"
        out.append(f"| {t} | {c} | {ns} | {gs} | {ve} | {vc} |")

    out += ["", "## Does the feedback harness change what the model writes?", "",
            "| condition | round | candidates | naive-correct | rigorous-correct | claimed | real (vs eager) |",
            "|---|---|---|---|---|---|---|"]
    for c in ("naive_feedback", "rigorous_feedback"):
        for rnd in sorted({r["round"] for r in recs}):
            rs = [r for r in ran if r["cond"] == c and r["round"] == rnd]
            out.append(f"| {c} | {rnd} | {len(rs)} | {sum(r['naive']['correct'] for r in rs)} | "
                       f"{sum(r['rigorous']['correct'] for r in rs)} | {sum(claimed(r) for r in rs)} | "
                       f"{sum(real(r) for r in rs)} |")
    for c in ("naive_feedback", "rigorous_feedback"):
        fin = [final[k] for k in final if k.endswith(c)]
        out.append(f"\nFinal kernels, {c}: real wins over eager {pct(sum(real(r) for r in fin), len(fin))}, "
                   f"over torch.compile {pct(sum(real(r, 'speedup_vs_compile') for r in fin), len(fin))}.")
    text = "\n".join(out) + "\n"
    (run / "report.md").write_text(text)
    return text


if __name__ == "__main__":
    print(main(Path(sys.argv[1])))
