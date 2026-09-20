import json,os,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parents[2];RUN=P/'artifacts/astra6_e30_expanded_CT_supervision_for_MR_20260910';PY=P.parent/'conda_envs/nnunet_v100/bin/python';EV=P/'.venv_official_eval_20260909/bin/python'
def progress(stage):
 p=RUN/'WORKFLOW_PROGRESS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'stage':stage,'timestamp':time.time(),'pid':os.getpid()})+'\n');os.replace(t,p);print('E30',stage,flush=True)
def main():
 progress('wait_existing_source_training')
 while not (RUN/'model/LOCKED.json').exists():time.sleep(30)
 for stage,exe,args in [('frozen_inference',PY,['scripts.astra6_e30.infer']),('official_six',EV,['scripts.analysis.current_official_evaluation','--version','E30']),('lesion_attribution',PY,['scripts.astra6_e30.evaluate']),('paired_comparison',EV,['scripts.astra6_e30.compare'])]:
  progress(stage);subprocess.run([str(exe),'-u','-m',*args],cwd=P,check=True)
 progress('complete_next_capability_review_required')
if __name__=='__main__':main()
