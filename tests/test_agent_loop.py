"""The agent loop end to end on CPU, against a stub OpenAI-compatible server.

The stub answers every request with two fixed files: an honest torch-only kernel
(no Triton, so the rigorous harness rejects it) and one that caches its output.
This checks the plumbing (generation, subprocess evaluation, logging, feedback,
the final selection and the report), not any model.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from kaa import agent, report

HONEST = "```python\nimport torch\ndef kernel(x):\n    return torch.softmax(x.float(), -1).to(x.dtype)\n```"
CACHING = ("```python\nimport torch\n_c = {}\ndef kernel(x):\n    if x.shape not in _c:\n"
           "        _c[x.shape] = torch.softmax(x.float(), -1).to(x.dtype)\n    return _c[x.shape]\n```")
SEEN = []


class Stub(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(body)
        choices = [{"message": {"content": [HONEST, CACHING][i % 2]}} for i in range(body["n"])]
        data = json.dumps({"choices": choices}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture()
def server():
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_loop(server, tmp_path, monkeypatch):
    SEEN.clear()
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["agent", "--base-url", server, "--model", "stub", "--tasks", "softmax",
                                      "--rounds", "2", "--samples", "2", "--out", str(out),
                                      "--baselines", str(tmp_path / "none.json")])
    agent.main()
    recs = [json.loads(ln) for ln in (out / "candidates.jsonl").read_text().splitlines()]
    assert len(recs) == 2 * 2 * 2                      # rounds x conditions x samples
    by = {(r["cond"], r["sample"]): r for r in recs if r["round"] == 0}
    honest, cache = by[("naive_feedback", 0)], by[("naive_feedback", 1)]
    assert honest["naive"]["correct"] and honest["rigorous"]["reason"] == "not_a_triton_kernel"
    assert cache["naive"]["correct"] and cache["rigorous"]["reason"] in ("not_a_triton_kernel", "stale_output")
    # Round 1 prompts carry round 0's feedback, in each condition's own words.
    r1 = [b for b in SEEN[2:]]
    texts = [b["messages"][-1]["content"] for b in r1]
    assert any("speed of the PyTorch reference" in t for t in texts)
    assert any("not_a_triton_kernel" in t for t in texts)
    final = json.loads((out / "final.json").read_text())
    assert set(final) == {"softmax|naive_feedback", "softmax|rigorous_feedback"}
    text = report.main(out)
    assert "Speedup claims" in text and "not_a_triton_kernel" in text


def test_loop_with_relative_out_dir(server, tmp_path, monkeypatch):
    """The output folder as a relative path, as tools/run_vm.sh passes it: every candidate
    must still be found and run (none may come back as an import failure)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["agent", "--base-url", server, "--model", "stub", "--tasks", "softmax",
                                      "--rounds", "1", "--samples", "2", "--out", "results/rel",
                                      "--baselines", "results/none.json"])
    agent.main()
    recs = [json.loads(ln) for ln in (tmp_path / "results/rel/candidates.jsonl").read_text().splitlines()]
    assert recs and all("naive" in r for r in recs), [r.get("load_error") for r in recs]
