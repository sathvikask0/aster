#!/usr/bin/env bash
# The 35M ladder, one rung per run, one variable per rung.
#
#   1  protein live, text frozen (MiniLM)      -- unfreezing the protein tower
#   2  protein live, text frozen (mpnet)       -- a bigger text encoder
#   3  protein live, text live   (mpnet)       -- training the text tower
#
# Each rung changes exactly one thing from the rung above, which is the only
# reason the deltas mean anything. Logs land beside the reports.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-6}"
mkdir -p reports logs

run () {
  local name="$1"; shift
  local out="reports/ft_35m_${name}_s${SEED}.json"
  local log="logs/ft_35m_${name}_s${SEED}.log"
  if [ -f "$out" ]; then
    echo "== skip ${name}: ${out} exists"
    return
  fi
  echo "== ${name} -> ${out}  (log: ${log})"
  uv run python scripts/run_amp_finetune.py \
    --esm 35M --trainable-blocks 2 \
    --negative-policy matched --epochs "$EPOCHS" --seed "$SEED" \
    --out "$out" "$@" 2>&1 | tee "$log"
}

run minilm_frozen
run mpnet_frozen  --text-model mpnet
run mpnet_live    --text-model mpnet --unfreeze-text --text-trainable-blocks 2

echo
echo "== comparison =="
uv run python scripts/compare_amp_runs.py reports/ft_35m_*_s"${SEED}".json \
  | tee "logs/compare_35m_s${SEED}.log"
