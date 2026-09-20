import os,json,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parents[2];RUN=P/'artifacts/astra6_e33_source_only_detector_20260910';D=P.parent/'conda_envs/nndet/bin/python';PY=P.parent/'conda_envs/nnunet_v100/bin/python';EV=P/'.venv_official_eval_20260909/bin/python'
def progress(stage):
 t=RUN/'WORKFLOW_PROGRESS.tmp';t.write_text(json.dumps({'stage':stage,'timestamp':time.time(),'pid':os.getpid()})+'\n');os.replace(t,RUN/'WORKFLOW_PROGRESS.json');print('E33 WORKFLOW',stage,flush=True)
def run(exe,module):subprocess.run([str(exe),'-u','-m',module],cwd=P,check=True)
def main():
 progress('waiting_source_preprocessing_and_E32_result')
 while not (RUN/'SOURCE_READY.json').exists() or not (P/'artifacts/astra6_e32_image_only_segmentation_20260910/DECISION.json').exists():time.sleep(30)
 progress('verified_local_cache');run(D,'scripts.astra6_e33.cache')
 progress('formal_training_45000_updates');run(D,'scripts.astra6_e33.train')
 progress('native_detector_inference');run(D,'scripts.astra6_e33.infer_detector')
 progress('fixed_C_S_F_native_inference');run(PY,'scripts.astra6_e33.infer')
 progress('official_six');subprocess.run([str(EV),'-u','-m','scripts.analysis.current_official_evaluation','--version','E33'],cwd=P,check=True)
 progress('candidate_and_lesion_attribution');run(PY,'scripts.astra6_e33.evaluate')
 progress('paired_official_comparison');run(EV,'scripts.astra6_e33.compare');progress('complete_next_capability_review_required')
if __name__=='__main__':main()
