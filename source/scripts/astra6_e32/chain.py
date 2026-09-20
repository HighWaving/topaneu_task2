import os,json,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parents[2];RUN=P/'artifacts/astra6_e32_image_only_segmentation_20260910';PY=P.parent/'conda_envs/nnunet_v100/bin/python';EV=P/'.venv_official_eval_20260909/bin/python'
def progress(stage):
 p=RUN/'WORKFLOW_PROGRESS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'stage':stage,'timestamp':time.time(),'pid':os.getpid()})+'\n');os.replace(t,p);print('E32 workflow',stage,flush=True)
def run(exe,module,*args):subprocess.run([str(exe),'-u','-m',module,*args],cwd=P,check=True)
def main():
 progress('full_formal_training');run(PY,'scripts.astra6_e32.train','--arm','normalized')
 progress('frozen_inference');run(PY,'scripts.astra6_e32.infer','--arm','normalized')
 progress('official_six');run(EV,'scripts.analysis.current_official_evaluation','--version','E32_normalized')
 progress('lesion_attribution');run(PY,'scripts.astra6_e32.evaluate','--arm','normalized')
 progress('native_entry_verification');run(PY,'scripts.astra6_e32.verify_native','--arm','normalized')
 progress('paired_comparison');run(EV,'scripts.astra6_e32.compare');progress('complete_next_capability_review_required')
if __name__=='__main__':main()
