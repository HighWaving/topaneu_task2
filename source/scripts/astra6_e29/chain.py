"""Continue the already-running joint control through both full E29 arms."""
import json
import os
import subprocess
import time
from scripts.astra6_e29.common import P, RUN, PYTHON, EVALUATOR_PYTHON, arm_path


def main():
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='GPU-a643ded3-193b-58e8-b362-be93dc8eac14',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    def stage(name):
        path = RUN / 'WORKFLOW_PROGRESS.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'stage': name, 'timestamp': time.time(), 'pid': os.getpid()}, indent=2) + '\n')
        tmp.replace(path)
        print('E29 workflow', name, flush=True)
    def run(module, *args, official=False):
        subprocess.run([str(EVALUATOR_PYTHON if official else PYTHON), '-m', module, *args],
                       cwd=P, env=env, check=True)
    stage('waiting_for_existing_joint_control_formal_training')
    begin = time.monotonic()
    while not (arm_path('joint') / 'model/LOCKED.json').exists():
        assert time.monotonic() - begin < 8 * 3600, 'Joint formal training incomplete after8h; inspect existing job, do not duplicate it'
        time.sleep(60)
    stage('staged_formal_anatomy_and_classification_training')
    run('scripts.astra6_e29.train', '--arm', 'staged')
    for arm in ['joint', 'staged']:
        stage(arm + '_fixed_upstream_inference')
        run('scripts.astra6_e29.infer', '--arm', arm)
        stage(arm + '_official_six')
        run('scripts.analysis.current_official_evaluation', '--version', 'E29_' + arm, official=True)
        stage(arm + '_paired_lesion_attribution')
        run('scripts.astra6_e29.evaluate', '--arm', arm)
    stage('paired_official_comparison_and_decision')
    run('scripts.astra6_e29.compare', official=True)
    stage('paired_results_complete_next_capability_review_required')


if __name__ == '__main__':
    main()
