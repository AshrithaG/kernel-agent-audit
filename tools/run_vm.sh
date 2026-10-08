#!/usr/bin/env bash
# The whole study on the GPU machine, in one go (about 2 hours, mostly kernel evaluation):
#
#   bash tools/run_vm.sh 2>&1 | tee ~/kaa.log
#
# STAGES picks a subset (default: tests baselines smoke agent report).
#   tests      the CPU tests (harness cheats, agent loop against a stub server)
#   baselines  eager and torch.compile timings of every reference, GPU otherwise idle
#   smoke      the real model on one task, one round, two samples: proves the pipeline
#   agent      every task, both feedback conditions, ROUNDS rounds of SAMPLES candidates
#   report     results/<run>/report.md
# MODEL (default Qwen/Qwen2.5-Coder-14B-Instruct-AWQ, 10 GB download on first run),
# ROUNDS (3), SAMPLES (4), RUN (results/run1) and PYTHON override the defaults.
set -euo pipefail
cd "$(dirname "$0")/.."

py="${PYTHON:-}"
if [[ -z "$py" ]]; then
  for cand in "$HOME"/*/.venv/bin/python "$HOME"/*-env/bin/python "$HOME"/.venv/bin/python; do
    [[ -x "$cand" ]] || continue
    if "$cand" -c 'import torch, triton, vllm; assert torch.cuda.is_available()' >/dev/null 2>&1; then
      py="$cand"; break
    fi
  done
fi
[[ -n "$py" ]] || { echo "No Python with torch, triton, vllm and a visible GPU; set PYTHON." >&2; exit 1; }
echo "python: $py"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || { echo "nvidia-smi failed: GPU unusable"; exit 1; }

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-14B-Instruct-AWQ}"
ROUNDS="${ROUNDS:-4}"; SAMPLES="${SAMPLES:-4}"; RUN="${RUN:-results/run3}"; PORT=8001
stages="${STAGES:-tests baselines smoke agent report}"
want() { [[ " $stages " == *" $1 "* ]]; }
step() { echo; echo "===== $* ($(date +%H:%M:%S))"; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo "other processes on the GPU (timings are only clean if this is empty):"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true

if want tests; then
  step tests
  pyt="$HOME/.cache/kaa-pytest"
  if "$py" -m pytest --version >/dev/null 2>&1; then "$py" -m pytest -q tests || echo "TESTS FAILED; continuing"
  elif "$py" -m pip install --quiet --target "$pyt" pytest >/dev/null 2>&1; then
    PYTHONPATH="$pyt:$PYTHONPATH" "$py" -m pytest -q tests || echo "TESTS FAILED; continuing"
  else echo "pytest unavailable; skipping tests"; fi
fi

if want baselines; then
  step "baselines (no model server running)"
  "$py" -m kaa.baselines --out results/baselines.json
fi

server_pid=""
start_server() {
  step "starting vLLM server for $MODEL"
  "$py" -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
    --gpu-memory-utilization 0.55 --max-model-len 8192 --enable-prefix-caching \
    > results/vllm_server.log 2>&1 &
  server_pid=$!
  for _ in $(seq 1 180); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "server up"; return 0; }
    kill -0 $server_pid 2>/dev/null || { echo "server died:"; tail -20 results/vllm_server.log; exit 1; }
    sleep 5
  done
  echo "server did not come up in 15 minutes"; tail -20 results/vllm_server.log; exit 1
}
stop_server() { [[ -n "$server_pid" ]] && kill $server_pid 2>/dev/null && wait $server_pid 2>/dev/null || true; }
trap stop_server EXIT

if want smoke || want agent; then start_server; fi

if want smoke; then
  step "smoke: one task, one round"
  rm -rf results/smoke
  "$py" -m kaa.agent --base-url "http://localhost:$PORT" --model "$MODEL" --tasks softmax \
    --rounds 1 --samples 2 --out results/smoke
  "$py" -m kaa.report results/smoke | head -20
  ran=$("$py" -c "import json; print(sum('naive' in json.loads(l) for l in open('results/smoke/candidates.jsonl')))")
  if [[ "$ran" == "0" ]]; then
    echo "SMOKE FAILED: no candidate ran. Stopping before the full study."; exit 1
  fi
  echo "smoke: $ran candidates ran"
fi

if want agent; then
  step "agent: all tasks, $ROUNDS rounds x $SAMPLES samples, both feedback conditions"
  "$py" -m kaa.agent --base-url "http://localhost:$PORT" --model "$MODEL" \
    --rounds "$ROUNDS" --samples "$SAMPLES" --out "$RUN"
fi
stop_server

if want report; then
  step report
  "$py" -m kaa.report "$RUN"
fi
