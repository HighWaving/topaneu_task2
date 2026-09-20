#!/usr/bin/env bash
# Polls MR fold1 and fold3 train.log until both reach 50 completed epochs
# (or their process exits), then writes a completion marker.
# 2026-08-17, peer-requested: notify when MR finishes, don't touch anything else.
set -uo pipefail

FOLD1_LOG="/home/jovyan/rtx4claude-datavol-1/nndet_models/Task030FG_TopAneuMR/RetinaUNetV001_D3V001_3d/fold1/train.log"
FOLD3_LOG="/home/jovyan/rtx4claude-datavol-1/nndet_models/Task030FG_TopAneuMR/RetinaUNetV001_D3V001_3d/fold3/train.log"
OUT="/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/logs/mr_folds_done.log"
TARGET=50

count_epochs() { grep -c "validation_epoch_end:268" "$1" 2>/dev/null || echo 0; }
proc_alive() { pgrep -f "nndet_train Task030FG_TopAneuMR -o exp.fold=$1" >/dev/null 2>&1; }

while true; do
  n1=$(count_epochs "$FOLD1_LOG")
  n3=$(count_epochs "$FOLD3_LOG")
  echo "[$(date -u +%H:%M:%S)] fold1=$n1/$TARGET fold3=$n3/$TARGET" >> "$OUT"
  fold1_done=0; fold3_done=0
  [ "$n1" -ge "$TARGET" ] || ! proc_alive 1 && fold1_done=1
  [ "$n3" -ge "$TARGET" ] || ! proc_alive 3 && fold3_done=1
  if [ "$fold1_done" -eq 1 ] && [ "$fold3_done" -eq 1 ]; then
    echo "[$(date -u +%H:%M:%S)] MR_FOLDS_DONE fold1=$n1 fold3=$n3" >> "$OUT"
    break
  fi
  sleep 300
done
