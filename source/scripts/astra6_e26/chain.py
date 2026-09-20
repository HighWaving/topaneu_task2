"""Complete the controlled training, source selection and gated official comparison."""
from pathlib import Path
import os,subprocess,time,json
P=Path(__file__).resolve().parents[2];R=P/'artifacts/astra6_e26_MR_segmentation_regularization_20260909';PY=P.parent/'conda_envs/nnunet_v100/bin/python';EV=P/'.venv_official_eval_20260909/bin/python'
def run(py,module,*args):subprocess.run([str(py),'-m',module,*args],cwd=P,env=env,check=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-f7491bbb-3972-1264-755a-63dec96b4a7d',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
if __name__=='__main__':
 while not (R/'native_validation/READY.json').exists():time.sleep(10)
 run(PY,'scripts.astra6_e26.train');lock=json.loads((R/'model/LOCKED.json').read_text())
 if lock['source_gate_passed']:
  run(PY,'scripts.astra6_e26.infer');run(EV,'scripts.analysis.current_official_evaluation','--version','E26');run(PY,'scripts.astra6_e26.diagnose');run(EV,'scripts.astra6_e26.compare')
 else:print('E26_SOURCE_GATE_FAILED_RETAIN_E17_NO_MR40',flush=True)
