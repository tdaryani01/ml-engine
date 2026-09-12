#!/usr/bin/env bash
# Background drawing-expert curriculum: gym batches → train → stop on val target.
# Usage:
#   ./examples/closed_loop_draw/run_expert_curriculum.sh
#   VAL_TARGET=0.75 MAX_ROUNDS=20 BATCH=16 ./examples/closed_loop_draw/run_expert_curriculum.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
TM_URI="${TM_URI:-http://127.0.0.1:8000}"
BATCH="${BATCH:-16}"
PRE="${PRE:-10}"
POST="${POST:-10}"
MAX_ROUNDS="${MAX_ROUNDS:-24}"
VAL_TARGET="${VAL_TARGET:-0.70}"
MIN_N_TRAIN="${MIN_N_TRAIN:-40}"
SEED0="${SEED0:-100}"
LOG="${LOG:-diagnostics_output/closed_loop_draw/expert_curriculum.log}"
mkdir -p "$(dirname "$LOG")"

echo "[curriculum] tm=$TM_URI batch=$BATCH rounds<=$MAX_ROUNDS val_target=$VAL_TARGET min_n=$MIN_N_TRAIN" | tee -a "$LOG"

round=0
best_val="nan"
while [[ "$round" -lt "$MAX_ROUNDS" ]]; do
  round=$((round + 1))
  seed=$((SEED0 + round * BATCH))
  echo "[curriculum] --- round $round seed=$seed ---" | tee -a "$LOG"
  PYTHONPATH=. .venv/bin/python examples/closed_loop_draw/expert_gym.py \
    --tm-uri "$TM_URI" \
    --instance-id expert-gym \
    --episodes "$BATCH" \
    --pre-traj "$PRE" \
    --post-traj "$POST" \
    --seed "$seed" \
    --train-after 2>&1 | tee -a "$LOG"

  # Pull latest train metrics from fleet heartbeat / last train line is in log;
  # also query episodes count via API.
  metrics="$(PYTHONPATH=. .venv/bin/python - <<PY
import json, urllib.request
base="$TM_URI".rstrip("/")
inst=json.loads(urllib.request.urlopen(base+"/api/instances", timeout=10).read())
hit=next((r for r in inst if r.get("id")=="drawing-expert"), {})
m=hit.get("metrics") or {}
print(json.dumps({
  "checkpoint_version": m.get("checkpoint_version"),
  "n_episodes": m.get("n_episodes"),
  "n_models": m.get("n_models"),
  "train_accuracy": m.get("train_accuracy"),
}))
PY
)"
  echo "[curriculum] fleet=$metrics" | tee -a "$LOG"

  # Parse last trained JSON from this round's log tail (val_accuracy if present).
  val="$(rg -o "\"val_accuracy\": [0-9.]+" "$LOG" | tail -1 | rg -o "[0-9.]+$" || true)"
  ntrain="$(rg -o "\"n_train\": [0-9]+" "$LOG" | tail -1 | rg -o "[0-9]+$" || true)"
  train_acc="$(rg -o "\"train_accuracy\": [0-9.]+" "$LOG" | tail -1 | rg -o "[0-9.]+$" || true)"
  echo "[curriculum] last_train n=$ntrain train_acc=$train_acc val_acc=$val" | tee -a "$LOG"

  if [[ -n "$val" && -n "$ntrain" ]]; then
    best_val="$val"
    # bc comparison
    ok="$(python3 - <<PY
n=int("$ntrain")
v=float("$val")
print(1 if (n >= int("$MIN_N_TRAIN") and v >= float("$VAL_TARGET")) else 0)
PY
)"
    if [[ "$ok" == "1" ]]; then
      echo "[curriculum] STOP decent val: n_train=$ntrain val_accuracy=$val >= $VAL_TARGET" | tee -a "$LOG"
      exit 0
    fi
  fi
done

echo "[curriculum] STOP max rounds=$MAX_ROUNDS best_val~=$best_val (target was $VAL_TARGET)" | tee -a "$LOG"
exit 0
