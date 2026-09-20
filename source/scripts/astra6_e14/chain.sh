#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/rtx4claude-datavol-1/topaneu2026_task2
export CUDA_VISIBLE_DEVICES=GPU-f7491bbb-3972-1264-755a-63dec96b4a7d OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/home/jovyan/rtx4claude-datavol-1/conda_envs/nnunet_v100/bin/python -m scripts.astra6_e14.train
/home/jovyan/rtx4claude-datavol-1/conda_envs/nnunet_v100/bin/python -m scripts.astra6_e14.infer
.venv_official_eval_20260909/bin/python scripts/analysis/current_official_evaluation.py --version E14
.venv_official_eval_20260909/bin/python -m scripts.astra6_e14.compare
/home/jovyan/rtx4claude-datavol-1/conda_envs/nnunet_v100/bin/python -m scripts.astra6_e14.diagnose
