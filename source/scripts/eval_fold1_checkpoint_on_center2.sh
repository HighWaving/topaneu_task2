#!/usr/bin/env bash
# Evaluates fold1's CURRENT checkpoint on a given center via pure box-vs-GT
# recall -- no HD95/masks, cheap and fast. 2026-08-14, peer-requested: track
# a real-model learning curve (mAP measures box regression quality, not "did
# the lesion make top-K", which is what we actually care about).
#
# center2 is fold1's held-out (clean generalization estimate, 40 cases).
# center1 is IN fold1's training set (peer-requested addition: quantifies
# the train-vs-held-out gap, i.e. overfitting, as a secondary signal given
# center2's held-out sample is thin).
#
# Snapshots the live model_last.ckpt to a timestamped copy first -- training
# writes to it continuously, and Lightning's own checkpoint writes are
# temp-file+rename (safe to copy mid-training), but a snapshot avoids any
# race with a NEW write landing between infer's read attempts.
set -euo pipefail

TAG="${1:?usage: eval_fold1_checkpoint_on_center2.sh <tag, e.g. epoch10> <center, e.g. center2>}"
CENTER="${2:?second arg: center id, e.g. center2 or center1}"

export det_data=/home/jovyan/rtx4claude-datavol-1/nndet_data
export det_models=/home/jovyan/rtx4claude-datavol-1/nndet_models
export OMP_NUM_THREADS=1
export PYTHONPATH="/home/jovyan/rtx4claude-datavol-1/new_aneurysms"

FOLD1_DIR="$det_models/Task030FG_TopAneuMR/RetinaUNetV001_D3V001_3d/fold1"
SNAPSHOT_DIR="/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/artifacts/fold1_checkpoint_snapshots/$TAG"
mkdir -p "$SNAPSHOT_DIR"
cp "$FOLD1_DIR/model_last.ckpt" "$SNAPSHOT_DIR/model_last.ckpt"
cp "$FOLD1_DIR/plan.pkl" "$SNAPSHOT_DIR/plan.pkl"
cp "$FOLD1_DIR/config.yaml" "$SNAPSHOT_DIR/config.yaml"
echo "[$(date -u +%H:%M:%S)] snapshotted checkpoint as $TAG"

cd /home/jovyan/rtx4claude-datavol-1/new_aneurysms
OUT_DIR="/home/jovyan/rtx4claude-datavol-1/topaneu2026_task2/artifacts/fold1_eval_${CENTER}_$TAG"
mkdir -p "$OUT_DIR"

CENTER_IDS=$(/home/jovyan/rtx4claude-datavol-1/conda_envs/topcow_claim_official/bin/python -c "
import json
m = json.loads(open('/home/jovyan/rtx4claude-datavol-1/topaneu2026_task1/artifacts/phase0/case_manifest.json').read())
ids = [c['case_id'] for c in m['cases'] if c['center']=='$CENTER' and c['modality']=='mr']
print(' '.join(f'--case-id {i}' for i in ids))
")

CUDA_VISIBLE_DEVICES=2 /home/jovyan/rtx4claude-datavol-1/conda_envs/nndet/bin/python \
  src/scripts/infer_nndet_adam.py \
  --training-dir "$SNAPSHOT_DIR" \
  --source-dir "$det_data/Task030FG_TopAneuMR/preprocessed/D3V001_3d/imagesTr" \
  --output-dir "$OUT_DIR" \
  --checkpoint last --device cuda:0 --num-tta 1 --max-detections 200 \
  $CENTER_IDS

echo "[$(date -u +%H:%M:%S)] FOLD1_EVAL_${CENTER}_${TAG}_DONE"
