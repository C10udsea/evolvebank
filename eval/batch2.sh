#!/bin/bash
# Eval phase: frozen bank (learned on train split) vs no bank, on unseen test tasks.
set -e
cd "$(dirname "$0")/.."
export $(grep -v '^#' .env | xargs)
export PYTHONUTF8=1 HF_ENDPOINT=https://hf-mirror.com

BANK=experiments/bank_learn80.json
[ -f "$BANK" ] || { echo "bank not found: $BANK"; exit 1; }

run_if_needed () {
  tag=$1; shift
  log=experiments/results/selfevolve_${tag}.jsonl
  if [ -f "$log" ] && [ "$(wc -l < "$log")" -ge 60 ]; then
    echo "=== $tag already complete, skipping ==="
  else
    echo "=== running/resuming $tag ==="
    .venv/bin/python eval/run_selfevolve.py --tag "$tag" --resume "$@"
  fi
}

for seed in 10 11; do
  run_if_needed eval_frozen_s${seed} --seed $seed --task-split test \
    --start-index 0 --end-index 60 --bank-mode topk --distill 0 --bank-path $BANK
  run_if_needed eval_none_s${seed} --seed $seed --task-split test \
    --start-index 0 --end-index 60 --bank-mode off --distill 0
done
echo "BATCH2 DONE"
