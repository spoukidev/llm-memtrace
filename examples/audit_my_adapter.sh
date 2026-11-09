#!/usr/bin/env bash
# Audit mode: score texts YOU own against YOUR adapter.
#   candidates.jsonl : {"text": ..., "label": 1|0 (optional)} per line, or a .txt with one text per line
#   nonmembers.txt   : texts you know were NOT in the fine-tune (used to calibrate thresholds / p-values)
set -euo pipefail
BASE=${BASE:-gpt2}
ADAPTER=${ADAPTER:-runs/gpt2-demo/target}
memtrace audit --base "$BASE" --adapter "$ADAPTER" \
  --candidates examples/my_candidates.jsonl \
  --nonmembers "${NONMEMBERS:-examples/nonmembers.txt}" \
  --out audits/mine
memtrace report --audit audits/mine/audit.json --out audits/mine/report
echo "open audits/mine/report/report.md"
