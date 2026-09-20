import os,json,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parents[2];RUN=P/'artifacts/astra6_e31_physical_candidate_segmentation_20260910';PY=P.parent/'conda_envs/nnunet_v100/bin/python';EV=P/'.venv_official_eval_20260909/bin/python'
def progress(stage):
 p=RUN/'WORKFLOW_PROGRESS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'stage':stage,'timestamp':time.time(),'pid':os.getpid()})+'\n');os.replace(t,p);print('E31 workflow',stage,flush=True)
def run(exe,module,*args):subprocess.run([str(exe),'-u','-m',module,*args],cwd=P,check=True)
def main():
 progress('wait_existing_crop_preparation')
 while not (RUN/'features/READY.json').exists():time.sleep(30)
 progress('native_source_validation_cache');run(PY,'scripts.astra6_e31.native_validation');run(PY,'scripts.astra6_e31.source_reference')
 progress('both_arms_full_formal_training')
 jobs=[]
 for arm,gpu in [('normalized','GPU-a643ded3-193b-58e8-b362-be93dc8eac14'),('physical','GPU-f7491bbb-3972-1264-755a-63dec96b4a7d')]:
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
  log=(RUN/(arm+'_train.log')).open('a')
  job=subprocess.Popen([str(PY),'-u','-m','scripts.astra6_e31.train','--arm',arm],cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT)
  jobs.append((arm,job,log))
 for arm,job,log in jobs:
  code=job.wait();log.close()
  if code:raise RuntimeError(f'{arm} training failed with exit {code}; inspect its persistent log. Other arm is not terminated.')
 for arm in ['normalized','physical']:
  progress(arm+'_frozen_inference');run(PY,'scripts.astra6_e31.infer','--arm',arm)
  progress(arm+'_official_six');run(EV,'scripts.analysis.current_official_evaluation','--version','E31_'+arm)
  progress(arm+'_lesion_attribution');run(PY,'scripts.astra6_e31.evaluate','--arm',arm)
 progress('paired_comparison');run(EV,'scripts.astra6_e31.compare');progress('complete_next_capability_review_required')
if __name__=='__main__':main()
