#!/usr/bin/env bash
# Chained pipeline: wait for image copy -> nndet_prep (Ts only, incremental) -> zero-shot predict.
# Peer-requested structural fix: && chaining instead of relying on Monitor notifications,
# so this makes progress even if nobody is watching. Each step's success gates the next;
# a failure stops the chain and leaves the error in the log, not a silent hang.
set -euo pipefail

COPY_PID="${1:?copy pid required}"

# Wait for the image copy into imagesTs to finish (process exit, not a fixed sleep).
while kill -0 "$COPY_PID" 2>/dev/null; do sleep 10; done
echo "[$(date -u +%H:%M:%S)] copy finished, starting nndet_prep"

export det_data=/home/jovyan/rtx4claude-datavol-1/nndet_data
export det_models=/home/jovyan/rtx4claude-datavol-1/nndet_models
export OMP_NUM_THREADS=1
export det_num_threads=8
export det_verbose=1

# Incremental: nndet_prep defaults to overwrite=False, so the 257 already-processed
# Task020FG training cases are skipped; only the new TopAneu-MR imagesTs cases run.
/home/jovyan/rtx4claude-datavol-1/conda_envs/nndet/bin/nndet_prep Task020FG_LocalAneurysm -np 16 -npp 8

echo "[$(date -u +%H:%M:%S)] nndet_prep done, starting zero-shot predict"

cd /home/jovyan/rtx4claude-datavol-1/new_aneurysms
export PYTHONPATH="/home/jovyan/rtx4claude-datavol-1/new_aneurysms${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p artifacts/nndet_task020_zeroshot_topaneu_mr logs/nndet_train_topaneu_mr

# No --case-id filter: predicts all of imagesTs (90 existing ADAM cases + 308
# new TopAneu-MR cases). Re-predicting ADAM is harmless waste (separate
# output-dir, doesn't touch existing ADAM results), and avoids a fragile
# 308-argument command line built via inline substitution.
CUDA_VISIBLE_DEVICES=2 /home/jovyan/rtx4claude-datavol-1/conda_envs/nndet/bin/python \
  src/scripts/infer_nndet_adam.py \
  --training-dir "$det_models/Task020FG_LocalAneurysm/RetinaUNetV001_D3V001_3d/fold0" \
  --source-dir "$det_data/Task020FG_LocalAneurysm/preprocessed/D3V001_3d/imagesTs" \
  --output-dir artifacts/nndet_task020_zeroshot_topaneu_mr \
  --checkpoint best --device cuda:0 --num-tta 1 --max-detections 200

echo "[$(date -u +%H:%M:%S)] ZEROSHOT_PREDICT_DONE"
