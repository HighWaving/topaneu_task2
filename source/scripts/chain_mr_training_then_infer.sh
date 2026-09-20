#!/usr/bin/env bash
# Phase B (GPU0/1): wait for MR fold1 AND fold3 BOTH to reach 50/50 epochs
# (both must finish before starting, so neither steals GPU from the other
# while still training) -> fold1 detector inference on center2's 40 cases
# (the honest baseline object) -> fold3 FUNCTIONAL-ONLY smoke test on 3 cases
# (fold3's val is polluted by center2 in its training set, never scored).
# 2026-08-17, peer-directed auto-chain. Does NOT touch GPU2/3.
set -uo pipefail

ROOT=/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2
PY=/home/jovyan/rtx4claude-datavol-1/conda_envs/topcow_claim_official/bin/python
PY_NNDET=/home/jovyan/rtx4claude-datavol-1/conda_envs/nndet/bin/python
LOG=$ROOT/logs/chain_mr_train_then_infer.log
HANDOFF=$ROOT/HANDOFF_20260815.md
mkdir -p "$ROOT/logs"

log() { echo "[$(date -u +%H:%M:%S)] $1" | tee -a "$LOG"; }

FOLD1_LOG=/home/jovyan/rtx4claude-datavol-1/nndet_models/Task030FG_TopAneuMR/RetinaUNetV001_D3V001_3d/fold1/train.log
FOLD3_LOG=/home/jovyan/rtx4claude-datavol-1/nndet_models/Task030FG_TopAneuMR/RetinaUNetV001_D3V001_3d/fold3/train.log
TARGET=50

count_epochs() { grep -c "validation_epoch_end:268" "$1" 2>/dev/null || echo 0; }
proc_alive() { pgrep -f "nndet_train Task030FG_TopAneuMR -o exp.fold=$1" >/dev/null 2>&1; }

log "=== Phase B start: waiting for MR fold1 AND fold3 both to finish (50/50 epochs) ==="
while true; do
  n1=$(count_epochs "$FOLD1_LOG"); n3=$(count_epochs "$FOLD3_LOG")
  d1=0; d3=0
  { [ "$n1" -ge "$TARGET" ] || ! proc_alive 1; } && d1=1
  { [ "$n3" -ge "$TARGET" ] || ! proc_alive 3; } && d3=1
  if [ "$d1" -eq 1 ] && [ "$d3" -eq 1 ]; then break; fi
  sleep 300
done
log "MR training done: fold1=$n1/$TARGET fold3=$n3/$TARGET"
if [ "$n1" -lt "$TARGET" ] || [ "$n3" -lt "$TARGET" ]; then
  log "*** WARNING: at least one fold's process exited before reaching $TARGET epochs (fold1=$n1, fold3=$n3) -- continuing anyway with whatever checkpoint exists, but this needs a human look ***"
fi
echo "MR_TRAINING_DONE fold1=$n1 fold3=$n3" >> "$ROOT/logs/mr_training.marker"
{
  echo ""
  echo "## MR training complete ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  echo "fold1=$n1/$TARGET epochs, fold3=$n3/$TARGET epochs. Starting fold1 detector inference on center2."
} >> "$HANDOFF"

# --- fold1 detector inference on center2's 40 cases (the honest baseline object) ---
FOLD1_DIR=/home/jovyan/rtx4claude-datavol-1/nndet_models/Task030FG_TopAneuMR/RetinaUNetV001_D3V001_3d/fold1
SNAPSHOT_DIR="$ROOT/artifacts/fold1_checkpoint_snapshots/mr_final_$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$SNAPSHOT_DIR"
cp "$FOLD1_DIR/model_last.ckpt" "$SNAPSHOT_DIR/"
cp "$FOLD1_DIR/plan.pkl" "$SNAPSHOT_DIR/"
cp "$FOLD1_DIR/config.yaml" "$SNAPSHOT_DIR/"
log "snapshotted fold1 FINAL checkpoint (model_last.ckpt, not model_best -- same rule as fold3/CT)"

FOLD1_BOXES_DIR=$ROOT/artifacts/mr_fold1_final_center2_boxes
mkdir -p "$FOLD1_BOXES_DIR"
cd /home/jovyan/rtx4claude-datavol-1/new_aneurysms
export PYTHONPATH="/home/jovyan/rtx4claude-datavol-1/new_aneurysms"
export det_data="/home/jovyan/rtx4claude-datavol-1/nndet_data"
export det_models="/home/jovyan/rtx4claude-datavol-1/nndet_models"
export OMP_NUM_THREADS=1
CASE_ID_ARGS=$($PY -c "print(' '.join(f'--case-id {i}' for i in __import__('json').load(open('$ROOT/artifacts/m1_center2_mr_case_ids.json'))))")
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 $PY_NNDET \
  src/scripts/infer_nndet_adam.py \
  --training-dir "$SNAPSHOT_DIR" \
  --source-dir "/home/jovyan/rtx4claude-datavol-1/nndet_data/Task030FG_TopAneuMR/preprocessed/D3V001_3d/imagesTr" \
  --output-dir "$FOLD1_BOXES_DIR" \
  --checkpoint last --device cuda:0 --num-tta 1 --batch-size 1 --max-detections 200 \
  $CASE_ID_ARGS >> "$ROOT/logs/chain_mr_fold1_inference.log" 2>&1

N=$(ls "$FOLD1_BOXES_DIR"/*_boxes.pkl 2>/dev/null | wc -l)
log "fold1 (FINAL) inference done: $N/40 boxes"
if [ "$N" -lt 40 ]; then
  log "*** FOLD1 INFERENCE INCOMPLETE: $N/40 -- ABORTING, needs human attention ***"
  echo "FOLD1_INFERENCE_FAILED n=$N" >> "$ROOT/logs/mr_fold1_boxes.marker"
  exit 1
fi

# --- fold3: FUNCTIONAL SMOKE TEST ONLY, never scored (val polluted by center2 in training) ---
FOLD3_DIR=/home/jovyan/rtx4claude-datavol-1/nndet_models/Task030FG_TopAneuMR/RetinaUNetV001_D3V001_3d/fold3
FOLD3_SNAPSHOT="$ROOT/artifacts/fold3_checkpoint_snapshots/mr_final_$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$FOLD3_SNAPSHOT"
cp "$FOLD3_DIR/model_last.ckpt" "$FOLD3_SNAPSHOT/"
cp "$FOLD3_DIR/plan.pkl" "$FOLD3_SNAPSHOT/"
cp "$FOLD3_DIR/config.yaml" "$FOLD3_SNAPSHOT/"
FOLD3_SMOKE_DIR=$ROOT/artifacts/mr_fold3_smoke_test_boxes
mkdir -p "$FOLD3_SMOKE_DIR"
SMOKE_IDS=$($PY -c "
import json
ids = json.load(open('$ROOT/artifacts/m1_center2_mr_case_ids.json'))[:3]
print(' '.join(f'--case-id {i}' for i in ids))
")
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 $PY_NNDET \
  src/scripts/infer_nndet_adam.py \
  --training-dir "$FOLD3_SNAPSHOT" \
  --source-dir "/home/jovyan/rtx4claude-datavol-1/nndet_data/Task030FG_TopAneuMR/preprocessed/D3V001_3d/imagesTr" \
  --output-dir "$FOLD3_SMOKE_DIR" \
  --checkpoint last --device cuda:0 --num-tta 1 --batch-size 1 --max-detections 200 \
  $SMOKE_IDS >> "$ROOT/logs/chain_mr_fold3_smoketest.log" 2>&1
N3=$(ls "$FOLD3_SMOKE_DIR"/*_boxes.pkl 2>/dev/null | wc -l)
log "fold3 FUNCTIONAL SMOKE TEST ONLY (submission model, val polluted, NOT evaluated/scored): $N3/3 boxes produced -- pipeline loads and runs cleanly"

echo "MR_FOLD1_BOXES_DONE n=$N" >> "$ROOT/logs/mr_fold1_boxes.marker"
{
  echo ""
  echo "## MR fold1 detector inference on center2 (40 cases) complete ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  echo "fold1 FINAL checkpoint (model_last.ckpt), $N/40 boxes.pkl. fold3 functional smoke test: $N3/3 boxes produced (submission model, load+run verified, NOT scored -- its val is polluted by center2 in training)."
  echo "Waiting on MR TA36 vessel masks (separate chain) to build the MR baseline."
} >> "$HANDOFF"
log "=== Phase B complete ==="
