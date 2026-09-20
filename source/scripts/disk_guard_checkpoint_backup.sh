#!/usr/bin/env bash
# Watches disk headroom; if it drops below 1.05TB, snapshots fold1/fold3's
# latest checkpoints to a safe location and logs it, so a later guardrail
# crash doesn't lose days of training. 2026-08-15, peer-requested.
set -uo pipefail

THRESHOLD_GB=1075  # 1.05 TB in GB (1024-based, matches df -BG)
HARD_STOP_GB=200    # below this: stop training gracefully instead of just backing up
SAFE_DIR="/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/artifacts/checkpoint_safety_backups"
LOG="/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/logs/disk_guard.log"
mkdir -p "$SAFE_DIR" "$(dirname "$LOG")"

already_backed_up=0
hard_stopped=0

# Training master PIDs, identified 2026-08-15 by cwd+cmdline (nndet_train, one per fold).
# If a PID has died/been replaced by the time the hard stop fires, re-resolve by cmdline match.
TRAIN_PIDS_FOLD1=1098685
TRAIN_PIDS_FOLD3=1134078
TRAIN_PIDS_CTFOLD2=1463526

resolve_pid() {
  # $1: last-known pid, $2: grep pattern to re-find it if it's gone (fold identity)
  if kill -0 "$1" 2>/dev/null; then
    echo "$1"
  else
    pgrep -f "$2" | head -1
  fi
}

while true; do
  avail_gb=$(df --output=avail -BG /home/jovyan/rtx4claude-datavol-1 | tail -1 | tr -dc '0-9')
  echo "[$(date -u +%H:%M:%S)] disk avail: ${avail_gb}GB" >> "$LOG"

  if [ "$avail_gb" -lt "$HARD_STOP_GB" ] && [ "$hard_stopped" -eq 0 ]; then
    echo "[$(date -u +%H:%M:%S)] HARD STOP: ${avail_gb}GB < ${HARD_STOP_GB}GB, stopping all three trainings gracefully" >> "$LOG"
    TS=$(date -u +%Y%m%dT%H%M%S)
    for FOLD in fold1 fold3 ctfold2; do
      SRC="/home/jovyan/rtx4claude-datavol-1/nndet_models/Task030FG_TopAneuMR/RetinaUNetV001_D3V001_3d/$FOLD"
      [ "$FOLD" = "ctfold2" ] && SRC="/home/jovyan/rtx4claude-datavol-1/nndet_models/Task031FG_TopAneuCT/RetinaUNetV001_D3V001_3d/fold2"
      DST="$SAFE_DIR/${FOLD}_hardstop_${TS}"
      mkdir -p "$DST"
      cp "$SRC/model_last.ckpt" "$DST/" 2>>"$LOG"
      cp "$SRC/plan.pkl" "$DST/" 2>>"$LOG"
      cp "$SRC/config.yaml" "$DST/" 2>>"$LOG"
      echo "[$(date -u +%H:%M:%S)] pre-stop backup of $FOLD to $DST" >> "$LOG"
    done
    PID1=$(resolve_pid "$TRAIN_PIDS_FOLD1" "nndet_train Task030FG_TopAneuMR -o exp.fold=1")
    PID3=$(resolve_pid "$TRAIN_PIDS_FOLD3" "nndet_train Task030FG_TopAneuMR -o exp.fold=3")
    PIDCT=$(resolve_pid "$TRAIN_PIDS_CTFOLD2" "nndet_train Task031FG_TopAneuCT -o exp.fold=2")
    for P in "$PID1" "$PID3" "$PIDCT"; do
      if [ -n "$P" ]; then
        echo "[$(date -u +%H:%M:%S)] SIGTERM -> pid $P" >> "$LOG"
        kill -TERM "$P" 2>>"$LOG"
      fi
    done
    hard_stopped=1
    echo "[$(date -u +%H:%M:%S)] HARD STOP complete. Trainings signalled to stop; checkpoints already backed up above." >> "$LOG"
  fi

  if [ "$avail_gb" -lt "$THRESHOLD_GB" ] && [ "$already_backed_up" -eq 0 ]; then
    TS=$(date -u +%Y%m%dT%H%M%S)
    echo "[$(date -u +%H:%M:%S)] GUARDRAIL: ${avail_gb}GB < ${THRESHOLD_GB}GB, backing up checkpoints" >> "$LOG"
    for FOLD in fold1 fold3; do
      SRC="/home/jovyan/rtx4claude-datavol-1/nndet_models/Task030FG_TopAneuMR/RetinaUNetV001_D3V001_3d/$FOLD"
      DST="$SAFE_DIR/${FOLD}_${TS}"
      mkdir -p "$DST"
      cp "$SRC/model_last.ckpt" "$DST/" 2>>"$LOG"
      cp "$SRC/plan.pkl" "$DST/" 2>>"$LOG"
      cp "$SRC/config.yaml" "$DST/" 2>>"$LOG"
      echo "[$(date -u +%H:%M:%S)] backed up $FOLD to $DST" >> "$LOG"
    done
    already_backed_up=1
  fi
  if [ "$avail_gb" -ge "$THRESHOLD_GB" ]; then
    already_backed_up=0  # re-arm if disk recovers, in case it drops again later
  fi
  sleep 600
done
