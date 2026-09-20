#!/usr/bin/env bash
# M1: fold1's CURRENT checkpoint x center2 (40 MR cases, honest LOCO position
# classifier) x the OFFICIAL evaluate.py. 2026-08-15, peer-directed mainline
# priority -- the single goal is a real, honest first end-to-end score.
set -euo pipefail

ROOT=/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2
DET_DATA=/home/jovyan/rtx4claude-datavol-1/nndet_data
DET_MODELS=/home/jovyan/rtx4claude-datavol-1/nndet_models
PY=/home/jovyan/rtx4claude-datavol-1/conda_envs/topcow_claim_official/bin/python
PY_NNDET=/home/jovyan/rtx4claude-datavol-1/conda_envs/nndet/bin/python

TAG="m1_$(date -u +%Y%m%dT%H%M%S)"
SNAPSHOT_DIR="$ROOT/artifacts/fold1_checkpoint_snapshots/$TAG"
BOXES_DIR="$ROOT/artifacts/m1_fold1_center2_boxes"
MASKS_DIR="$ROOT/artifacts/m1_fold1_center2_masks"
CASE_IDS_JSON="$ROOT/artifacts/m1_center2_mr_case_ids.json"
mkdir -p "$SNAPSHOT_DIR" "$BOXES_DIR" "$MASKS_DIR" "$ROOT/reports" "$ROOT/logs"

echo "[$(date -u +%H:%M:%S)] === M1 start, tag=$TAG ===" | tee -a "$ROOT/logs/run_m1.log"

# 1. Snapshot fold1's CURRENT live checkpoint (whatever epoch it's at right now).
FOLD1_DIR="$DET_MODELS/Task030FG_TopAneuMR/RetinaUNetV001_D3V001_3d/fold1"
cp "$FOLD1_DIR/model_last.ckpt" "$SNAPSHOT_DIR/model_last.ckpt"
cp "$FOLD1_DIR/plan.pkl" "$SNAPSHOT_DIR/plan.pkl"
cp "$FOLD1_DIR/config.yaml" "$SNAPSHOT_DIR/config.yaml"
echo "[$(date -u +%H:%M:%S)] snapshotted fold1 checkpoint as $TAG" | tee -a "$ROOT/logs/run_m1.log"

# 2. Case list: center2 MR only (40 cases) -- NOT loco_splits.json's mixed-
# modality center2 fold (87 cases incl. CT, which fold1's MR-only detector
# was never meant to see), and NOT center5 (in fold1's training set).
$PY -c "
import json
m = json.loads(open('/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/artifacts/phase0/case_manifest.json').read())
ids = sorted(c['case_id'] for c in m['cases'] if c['center']=='center2' and c['modality']=='mr')
json.dump(ids, open('$CASE_IDS_JSON', 'w'), indent=2)
print(f'{len(ids)} center2 MR case ids -> $CASE_IDS_JSON')
"

# 3. Inference on GPU2 (best available headroom), batch-size 1 for OOM
# safety (epoch10's default batch=4 OOM'd here once already), under the
# file-access audit wrapper (no strace binary in this environment).
echo "[$(date -u +%H:%M:%S)] running inference..." | tee -a "$ROOT/logs/run_m1.log"
cd /home/jovyan/rtx4claude-datavol-1/new_aneurysms
export PYTHONPATH="/home/jovyan/rtx4claude-datavol-1/new_aneurysms"
export det_data="$DET_DATA"
export det_models="$DET_MODELS"
export OMP_NUM_THREADS=1
CASE_ID_ARGS=$($PY -c "print(' '.join(f'--case-id {i}' for i in __import__('json').load(open('$CASE_IDS_JSON'))))")
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 $PY_NNDET \
  "$ROOT/scripts/audit_wrapper.py" \
  src/scripts/infer_nndet_adam.py \
  --training-dir "$SNAPSHOT_DIR" \
  --source-dir "$DET_DATA/Task030FG_TopAneuMR/preprocessed/D3V001_3d/imagesTr" \
  --output-dir "$BOXES_DIR" \
  --checkpoint last --device cuda:0 --num-tta 1 --batch-size 1 --max-detections 200 \
  $CASE_ID_ARGS \
  2>&1 | tee -a "$ROOT/logs/run_m1_inference.log"

N_BOXES=$(ls "$BOXES_DIR"/*_boxes.pkl 2>/dev/null | wc -l)
echo "[$(date -u +%H:%M:%S)] inference done: $N_BOXES/40 boxes.pkl written" | tee -a "$ROOT/logs/run_m1.log"
if [ "$N_BOXES" -eq 0 ]; then
  echo "[$(date -u +%H:%M:%S)] *** M1 FAILED: zero boxes written, inference crashed ***" | tee -a "$ROOT/logs/run_m1.log"
  exit 1
fi

# 4. Build 0-52 masks with the HONEST, axis-corrected position classifier
# (LOCO: trained on center1/4/5 only, center2 held out), under the same audit.
echo "[$(date -u +%H:%M:%S)] building masks..." | tee -a "$ROOT/logs/run_m1.log"
cd "$ROOT"
$PY "$ROOT/scripts/audit_wrapper.py" \
  "$ROOT/scripts/build_m1_masks.py" \
  --boxes-dir "$BOXES_DIR" --output-dir "$MASKS_DIR" --held-out-center center2 \
  2>&1 | tee -a "$ROOT/logs/run_m1_maskbuild.log"

# 5. Lesion-level diagnostics (sensitivity, FN, FP/case, candidates/case) --
# informational, NOT one of the official 6 metrics.
echo "[$(date -u +%H:%M:%S)] lesion diagnostics..." | tee -a "$ROOT/logs/run_m1.log"
$PY "$ROOT/scripts/m1_lesion_diagnostics.py" \
  --boxes-dir "$BOXES_DIR" --case-ids-json "$CASE_IDS_JSON" \
  --output "$ROOT/reports/m1_lesion_diagnostics.json" \
  2>&1 | tee -a "$ROOT/logs/run_m1_lesion_diag.log"

# 6. Official scoring (evaluate.py, wrapped, unmodified).
echo "[$(date -u +%H:%M:%S)] official scoring..." | tee -a "$ROOT/logs/run_m1.log"
$PY "$ROOT/scripts/score_m1.py" \
  --predictions-dir "$MASKS_DIR" --case-ids-json "$CASE_IDS_JSON" \
  --output "$ROOT/reports/m1_official_score.json" \
  2>&1 | tee -a "$ROOT/logs/run_m1_score.log"

echo "[$(date -u +%H:%M:%S)] === M1_PIPELINE_DONE tag=$TAG ===" | tee -a "$ROOT/logs/run_m1.log"
