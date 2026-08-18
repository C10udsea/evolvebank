#!/bin/bash
# Formal experiment batch: self-evolving (topk) vs control (off), 2 seeds each.
set -e
cd "$(dirname "$0")/.."
export $(grep -v '^#' .env | xargs)
export PYTHONUTF8=1 HF_ENDPOINT=https://hf-mirror.com

run_if_needed () {  # $1 = tag, rest = args
  tag=$1; shift
  log=experiments/results/selfevolve_${tag}.jsonl
  if [ -f "$log" ] && [ "$(wc -l < "$log")" -ge 30 ]; then
    echo "=== $tag already complete, skipping ==="
  else
    rm -f "$log" experiments/bank_${tag}.json
    echo "=== running $tag ==="
    .venv/bin/python eval/run_selfevolve.py --tag "$tag" "$@"
  fi
}

for seed in 10 11; do
  run_if_needed se_topk_s${seed} --seed $seed --start-index 0 --end-index 30 --bank-mode topk --sources both
  run_if_needed se_off_s${seed} --seed $seed --start-index 0 --end-index 30 --bank-mode off --sources both
done
echo "BATCH DONE"
