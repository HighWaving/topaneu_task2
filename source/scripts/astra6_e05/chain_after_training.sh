#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/rtx4claude-datavol-1/topaneu2026_task2
run_dir=artifacts/astra6_e05_learned_location_splits_20260909
if [ ! -f "$run_dir/model/CLASSIFIER_LOCKED.json" ]; then
  echo 'Formal training is not yet locked; run only after training completes.' >&2
  exit 1
fi
if [ ! -f "$run_dir/PREDICTIONS_HASHED_BEFORE_EVAL_GT.json" ]; then
  CUDA_VISIBLE_DEVICES=GPU-a643ded3-193b-58e8-b362-be93dc8eac14 OMP_NUM_THREADS=4 /home/jovyan/rtx4claude-datavol-1/conda_envs/nnunet_v100/bin/python -m scripts.astra6_e05.infer_e05
fi
.venv_official_eval_20260909/bin/python scripts/analysis/current_official_evaluation.py --version E05
.venv_official_eval_20260909/bin/python -m scripts.astra6_e05.compare_e05
OMP_NUM_THREADS=4 /home/jovyan/rtx4claude-datavol-1/conda_envs/nnunet_v100/bin/python -m scripts.astra6_e05.diagnose_e05
