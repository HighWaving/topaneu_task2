#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/rtx4claude-datavol-1/topaneu2026_task2
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/jovyan/rtx4claude-datavol-1/conda_envs/nnunet_v100/bin/python -m scripts.astra6_e09.train
CUDA_VISIBLE_DEVICES=GPU-a643ded3-193b-58e8-b362-be93dc8eac14 OMP_NUM_THREADS=4 /home/jovyan/rtx4claude-datavol-1/conda_envs/nnunet_v100/bin/python -m scripts.astra6_e09.infer
.venv_official_eval_20260909/bin/python scripts/analysis/current_official_evaluation.py --version CT_E09
.venv_official_eval_20260909/bin/python -m scripts.astra6_e09.compare
