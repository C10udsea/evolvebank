#!/bin/bash
# Final eval completion: unique-task-id guard (line counts can be inflated
# by duplicate lines from interrupted resumes).
set -e
cd "$(dirname "$0")/.."
export $(grep -v '^#' .env | xargs)
export PYTHONUTF8=1 HF_ENDPOINT=https://hf-mirror.com

BANK=experiments/bank_learn80.json
[ -f "$BANK" ] || { echo "bank not found: $BANK"; exit 1; }

unique_done () {
  .venv/bin/python - "$1" <<'PY'
import json, sys
try:
    print(len({json.loads(l)["task_id"] for l in open(sys.argv[1], encoding="utf-8")}))
except FileNotFoundError:
    print(0)
PY
}

run_if_needed () {
  tag=$1; shift
  log=experiments/results/selfevolve_${tag}.jsonl
  n=$(unique_done "$log")
  if [ "$n" -ge 60 ]; then
    echo "=== $tag complete ($n/60 unique), skipping ==="
  else
    echo "=== running/resuming $tag ($n/60 unique) ==="
    .venv/bin/python eval/run_selfevolve.py --tag "$tag" --resume "$@"
  fi
}

for seed in 10 11; do
  run_if_needed eval_frozen_s${seed} --seed $seed --task-split test \
    --start-index 0 --end-index 60 --bank-mode topk --distill 0 --bank-path $BANK
  run_if_needed eval_none_s${seed} --seed $seed --task-split test \
    --start-index 0 --end-index 60 --bank-mode off --distill 0
done
echo "BATCH3 DONE"
