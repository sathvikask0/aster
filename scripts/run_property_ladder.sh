#!/usr/bin/env bash
# The property ladder: train on antimicrobial, test on hemolytic.
#
# This is the first benchmark in the project where every negative carries a
# measured concentration and the two tasks genuinely disagree (46% of the 897
# peptides measured for both). So it is the first run whose lift over the
# ceiling can be read as evidence about the premise rather than about a shortcut.
#
# Read it in this order:
#   entity_only_*  never sees the question. If it matches dual, the question is
#                  not what is doing the work and there is no transfer claim.
#   *_frozen       the same heads over cached vectors. live minus frozen is the
#                  value of unfreezing, and it is a within-seed comparison.
#   mechanism swap accuracy under the wrong property's prompt. If it survives,
#                  the prompt is a task id spelled in English.
#   task_id_frozen degenerate here: one training task means one id. Not a control.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-42 43 44}"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ESM="${ESM:-35M}"
DEVICE="${DEVICE:-mps}"
HELD_OUT="${HELD_OUT:-hemolytic}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p reports logs

for seed in $SEEDS; do
  name="prop_${ESM}_${HELD_OUT}_e${EPOCHS}_b${BATCH_SIZE}_s${seed}"
  out="reports/${name}.json"
  if [[ -e "$out" ]]; then
    echo "Report already exists: $out; inspect or rename it before rerunning."
    exit 1
  fi
  uv run --locked python -u scripts/run_amp_finetune.py \
    --benchmark properties --held-out-property "$HELD_OUT" \
    --esm "$ESM" --trainable-blocks 2 \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --seed "$seed" --device "$DEVICE" --out "$out" \
    2>&1 | tee "logs/${name}.log"
done

echo
echo "Compare the seeds:"
echo "  uv run python scripts/compare_amp_runs.py reports/prop_${ESM}_${HELD_OUT}_e${EPOCHS}_b${BATCH_SIZE}_s*.json"
