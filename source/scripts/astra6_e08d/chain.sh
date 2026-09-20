#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/rtx4claude-datavol-1/topaneu2026_task2
/home/jovyan/rtx4claude-datavol-1/conda_envs/nnunet_v100/bin/python -m scripts.astra6_e08d.setup
CUDA_VISIBLE_DEVICES=GPU-f7491bbb-3972-1264-755a-63dec96b4a7d OMP_NUM_THREADS=4 /home/jovyan/rtx4claude-datavol-1/conda_envs/nnunet_v100/bin/python -m scripts.astra6_e08d.infer
.venv_official_eval_20260909/bin/python scripts/analysis/current_official_evaluation.py --version E08D
.venv_official_eval_20260909/bin/python -m scripts.astra6_e08d.compare
OMP_NUM_THREADS=4 /home/jovyan/rtx4claude-datavol-1/conda_envs/nnunet_v100/bin/python -m scripts.astra6_e08d.diagnose
