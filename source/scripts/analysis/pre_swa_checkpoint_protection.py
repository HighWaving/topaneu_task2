"""One-time pre-SWA archive triggers inside the existing five-minute monitor."""
import json,os,subprocess,datetime
from pathlib import Path
P=Path(__file__).resolve().parents[2];R=P/'artifacts/astra6_e23_MR_oof_candidates_20260909'
def snapshot():
 states={}
 for fold in [0,1]:
  lock=R/f'checkpoints/pre_swa/fold{fold}.json';progress=R/f'logs/fold{fold}_batch_progress.json'
  if lock.exists():
   states[str(fold)]={'status':'protected','record':json.loads(lock.read_text())};continue
  if not progress.exists():states[str(fold)]={'status':'waiting_for_training'};continue
  step=json.loads(progress.read_text())['global_step'];candidate=R/f'checkpoints/pre_swa/fold{fold}.candidate.ckpt'
  if step<36000:states[str(fold)]={'status':'waiting_before_pre_SWA_window','step':step};continue
  if step>=37500 and not candidate.exists():states[str(fold)]={'status':'MISSED_ARCHIVE_WINDOW_do_not_blindly_resume_legacy_SWA','step':step};continue
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
  cmd=[str(P.parent/'conda_envs/nndet/bin/python'),'-m','scripts.astra6_e23.safe_swa_resume','--snapshot','--fold',str(fold)]
  try:
   result=subprocess.run(cmd,cwd=P,env=env,capture_output=True,text=True,timeout=240)
   states[str(fold)]={'status':'protected' if result.returncode==0 else 'retry_archive_next_monitor_tick','step_at_start':step,'returncode':result.returncode,'stdout':result.stdout[-3000:],'stderr':result.stderr[-3000:]}
  except subprocess.TimeoutExpired:states[str(fold)]={'status':'archive_attempt_timeout_retry_next_tick','step':step}
 out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'folds':states,'healthy_trainers_unchanged':True,'no_additional_background_observer':True};dest=R/'checkpoints/PRE_SWA_ARCHIVE_STATUS.json';tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');os.replace(tmp,dest);return out
if __name__=='__main__':print(json.dumps(snapshot(),indent=2),flush=True)
