from pathlib import Path

P = Path(__file__).resolve().parents[2]
RUN = P / 'artifacts/astra6_e29_staged_anatomy_location_20260910'
PREVIOUS = P / 'artifacts/astra6_e28_joint_anatomy_location_20260909'
SEED = 20260910
PYTHON = P.parent / 'conda_envs/nnunet_v100/bin/python'
EVALUATOR_PYTHON = P / '.venv_official_eval_20260909/bin/python'


def arm_path(name):
    assert name in ['joint', 'staged']
    return RUN / name
