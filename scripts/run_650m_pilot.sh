#!/usr/bin/env bash
# Local capacity/learning pilot. Repeat promising results with SEEDS="42 43 44".
# MAX_PEPTIDES=0 selects the full benchmark rather than the default subset.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-42}"
MAX_PEPTIDES="${MAX_PEPTIDES:-2000}"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p reports logs

for seed in $SEEDS; do
  name="ft_650m_p${MAX_PEPTIDES}_e${EPOCHS}_b${BATCH_SIZE}_s${seed}_v2"
  out="reports/${name}.json"
  if [[ -e "$out" ]]; then
    echo "Report already exists: $out; choose new settings or inspect it before rerunning."
    exit 1
  fi
  uv run --locked python -u scripts/run_amp_finetune.py \
    --esm 650M --trainable-blocks 2 --cache-frozen-prefix \
    --text-model minilm --negative-policy matched \
    --max-peptides "$MAX_PEPTIDES" --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" --seed "$seed" --device mps \
    --checkpoint-dir "checkpoints/${name}" --out "$out" \
    2>&1 | tee "logs/${name}.log"
done
