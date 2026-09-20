"""Read-only persistent progress for the ongoing competitive-model work."""
from pathlib import Path
import json,time,datetime,os
P=Path(__file__).resolve().parents[2];D=P/'artifacts/astra6_e23_MR_oof_candidates_20260909';F=P/'artifacts/astra6_e25_oof_shape_fp_filter_20260909';S=P/'artifacts/astra6_e26_MR_segmentation_regularization_20260909';C=P/'artifacts/astra6_e27_junction_location_20260909'
C28=P/'artifacts/astra6_e28_joint_anatomy_location_20260909'
def read(path):
 try:return json.loads(path.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None
def main():
 from scripts.astra6_e23.monitor import snapshot as detector_snapshot
 from scripts.analysis.research_budget import snapshot as budget_snapshot
 from scripts.analysis.pre_swa_checkpoint_protection import snapshot as protect_pre_swa
 from scripts.analysis.e28_budget import snapshot as e28_budget_snapshot
 while True:
  detector_snapshot()
  protect_pre_swa()
  budget_snapshot()
  e28_budget_snapshot()
  from scripts.analysis.active_research_progress import snapshot as active_research_snapshot
  folds={}
  for f in [0,1]:
   batch=read(D/f'logs/fold{f}_batch_progress.json');trained=read(D/f'checkpoints/fold{f}_TRAINED.json');oof=read(D/f'checkpoints/fold{f}_OOF_COMPLETE.json');folds[str(f)]={'batch':batch,'last_batch_age_seconds':time.time()-batch['timestamp'] if batch else None,'planned_updates':45000,'training_complete':trained is not None,'OOF_complete':oof is not None,'checkpoint':trained,'OOF_n_cases':len(oof['cases']) if oof else None}
  f=read(F/'model/LOCKED.json');data={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'monitor_pid':os.getpid(),'pre_SWA_checkpoint_protection':read(D/'checkpoints/PRE_SWA_ARCHIVE_STATUS.json'),'best_release':read(P/'LATEST_MODEL_RELEASE.json'),'detector_source_folds':folds,'E25':{'source_ready':(F/'features/READY.json').exists(),'trained_model':f,'paired_evaluation_complete':(F/'evaluation/paired_official_comparison.json').exists()},'E26':{'training_complete':(S/'model/LOCKED.json').exists(),'official_complete':(P/'artifacts/current_official_20260909/E26/DONE.json').exists(),'decision':(read(S/'evaluation/DECISION.json') or {}).get('decision','official negative; paired analysis finishing')},'E27':{'training_complete':(C/'model/LOCKED.json').exists(),'decision':(read(C/'evaluation/DECISION.json') or {}).get('decision')},'E28':{'research_started':(C28/'RESEARCH_START.json').exists(),'source_ready':(C28/'features/READY.json').exists(),'development_latest':(read(C28/'model/development_history.json') or [None])[-1],'final_latest':(read(C28/'model/final_history.json') or [None])[-1],'decision':read(C28/'evaluation/DECISION.json'),'bottleneck_review':read(C28/'NEXT_BOTTLENECK_REVIEW_REQUIRED.json')},'platform_submission':'not_verified; team/algorithm and user-built container pending','interpretation':'Old batch timestamps can indicate planned validation; inspect the workflow log before diagnosing a stalled training run.'};path=P/'COMPETITIVE_LIVE_PROGRESS.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,path)
  active=active_research_snapshot()
  if active is not None:
   data['active_research']=active;tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,path)
  if (P/'RESEARCH_SESSION_STOP.json').exists():return
  time.sleep(300)
if __name__=='__main__':main()
