#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/rtx4claude-datavol-1/topaneu2026_task2
run_dir=artifacts/astra6_e06_image_fp_filter_20260909
if [ ! -f "$run_dir/model/LOCKED.json" ]; then
 CUDA_VISIBLE_DEVICES=GPU-f7491bbb-3972-1264-755a-63dec96b4a7d OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/jovyan/rtx4claude-datavol-1/conda_envs/nnunet_v100/bin/python -m scripts.astra6_e06.train
fi
if [ ! -f "$run_dir/PREDICTIONS_HASHED_BEFORE_EVAL_GT.json" ]; then
 CUDA_VISIBLE_DEVICES=GPU-f7491bbb-3972-1264-755a-63dec96b4a7d OMP_NUM_THREADS=4 /home/jovyan/rtx4claude-datavol-1/conda_envs/nnunet_v100/bin/python -m scripts.astra6_e06.infer
fi
.venv_official_eval_20260909/bin/python scripts/analysis/current_official_evaluation.py --version E06
.venv_official_eval_20260909/bin/python -m scripts.astra6_e06.compare
OMP_NUM_THREADS=4 /home/jovyan/rtx4claude-datavol-1/conda_envs/nnunet_v100/bin/python -m scripts.astra6_e06.diagnose
