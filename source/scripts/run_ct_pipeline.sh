#!/usr/bin/env bash
# Chained CT pipeline: nndet_prep -> nndet_unpack -> nndet_train (fold 2, full-data).
# 2026-08-14, peer-requested: mirror the MR line's process, chained with &&
# (not Monitor-dependent), with an explicit disk guardrail check before the
# expensive unpack step (peer's estimate: ~35-40GB for CT's 109 cases).
set -euo pipefail

export det_data=/home/jovyan/rtx4claude-datavol-1/nndet_data
export det_models=/home/jovyan/rtx4claude-datavol-1/nndet_models
export OMP_NUM_THREADS=1
export det_num_threads=8
export det_verbose=1

echo "[$(date -u +%H:%M:%S)] starting nndet_prep Task031FG_TopAneuCT"
/home/jovyan/rtx4claude-datavol-1/conda_envs/nndet/bin/nndet_prep Task031FG_TopAneuCT -np 16 -npp 8

avail_tb=$(df --output=avail -BG /home/jovyan/rtx4claude-datavol-1 | tail -1 | tr -dc '0-9')
echo "[$(date -u +%H:%M:%S)] nndet_prep done, disk avail: ${avail_tb}GB"
if [ "$avail_tb" -lt 1000 ]; then
  echo "[$(date -u +%H:%M:%S)] GUARDRAIL: disk below 1.0TB (${avail_tb}GB), stopping before unpack"
  exit 1
fi

echo "[$(date -u +%H:%M:%S)] starting nndet_unpack"
/home/jovyan/rtx4claude-datavol-1/conda_envs/nndet/bin/nndet_unpack \
  "$det_data/Task031FG_TopAneuCT/preprocessed/D3V001_3d/imagesTr" 16

avail_tb2=$(df --output=avail -BG /home/jovyan/rtx4claude-datavol-1 | tail -1 | tr -dc '0-9')
echo "[$(date -u +%H:%M:%S)] unpack done, disk avail: ${avail_tb2}GB"
if [ "$avail_tb2" -lt 1000 ]; then
  echo "[$(date -u +%H:%M:%S)] GUARDRAIL: disk below 1.0TB (${avail_tb2}GB) after unpack, stopping before train"
  exit 1
fi

echo "[$(date -u +%H:%M:%S)] starting nndet_train fold 2 (full-data submission model) on GPU3"
mkdir -p /home/jovyan/rtx4claude-datavol-1/new_aneurysms/logs/nndet_train_topaneu_ct
CUDA_VISIBLE_DEVICES=3 nohup /home/jovyan/rtx4claude-datavol-1/conda_envs/nndet/bin/nndet_train \
  Task031FG_TopAneuCT -o exp.fold=2 \
  > /home/jovyan/rtx4claude-datavol-1/new_aneurysms/logs/nndet_train_topaneu_ct/fold2.log 2>&1 &
TRAIN_PID=$!
echo "[$(date -u +%H:%M:%S)] CT_PIPELINE_TRAIN_LAUNCHED pid=$TRAIN_PID"
sleep 20
if kill -0 "$TRAIN_PID" 2>/dev/null; then
  echo "[$(date -u +%H:%M:%S)] confirmed alive after 20s"
else
  echo "[$(date -u +%H:%M:%S)] CRASHED within 20s -- check fold2.log"
  exit 1
fi
