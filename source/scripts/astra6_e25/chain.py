"""Consume completed E23 folds and proceed through complete F training."""
from pathlib import Path
import time,json,subprocess,os
P=Path(__file__).resolve().parents[2];R=P/'artifacts/astra6_e25_oof_shape_fp_filter_20260909';O=P/'artifacts/astra6_e23_MR_oof_candidates_20260909';PY=P.parent/'conda_envs/nnunet_v100/bin/python'
def run(module,*args):subprocess.run([str(PY),'-m',module,*args],cwd=P,env=env,check=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-a643ded3-193b-58e8-b362-be93dc8eac14',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
if __name__=='__main__':
 for fold in [0,1]:
  start=time.monotonic()
  while not (O/f'checkpoints/fold{fold}_OOF_COMPLETE.json').exists():
   assert time.monotonic()-start<48*3600,'Required E23 OOF fold not complete within48hours';time.sleep(300)
  # GPU2 is free after fold0 postprocessing completes. Fold1 may use GPU3 meanwhile.
  run('scripts.astra6_e25.prepare','--fold',str(fold))
 if not (R/'features/READY.json').exists():run('scripts.astra6_e25.prepare','--merge')
 run('scripts.astra6_e25.train')
 lock=json.loads((R/'model/LOCKED.json').read_text())
 run('scripts.astra6_e25.infer')
 subprocess.run([str(P/'.venv_official_eval_20260909/bin/python'),'-m','scripts.analysis.current_official_evaluation','--version','E25'],cwd=P,env=env,check=True)
 run('scripts.astra6_e25.diagnose')
 subprocess.run([str(P/'.venv_official_eval_20260909/bin/python'),'-m','scripts.astra6_e25.compare'],cwd=P,env=env,check=True)
