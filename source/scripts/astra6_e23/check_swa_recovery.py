"""Transaction checks on temporary tiny checkpoints; never fit/restart a model."""
from pathlib import Path
import tempfile,json,torch
from scripts.astra6_e23.safe_swa_resume import snapshot,prepare_resume,last_path,sha,copy_verified,RUN,P,active

def save(run,fold,epoch,step,weight):
 f=last_path(fold,run);f.parent.mkdir(parents=True,exist_ok=True);torch.save({'epoch':epoch,'global_step':step,'state_dict':{'weight':torch.tensor([weight])},'optimizer_states':[{'sentinel':123}],'lr_schedulers':[{'sentinel':456}]},f);return f

def main():
 checks={}
 with tempfile.TemporaryDirectory(prefix='e23_swa_recovery_check_') as td:
  run=Path(td);last=save(run,0,49,36750,1.);original=sha(last);rec=snapshot(0,run);assert rec['sha256']==original
  post=save(run,0,53,39750,2.);interrupted=sha(post);result=prepare_resume(0,run);assert sha(last)==original and sha(Path(result['interrupted_backup']))==interrupted;state=torch.load(last,map_location='cpu');assert state['optimizer_states']==[{'sentinel':123}] and state['lr_schedulers']==[{'sentinel':456}];checks['SWA_recovery_preserves_interrupted_and_restores_full_pre_SWA_checkpoint']=True
  assert prepare_resume(0,run)['action']=='ordinary_pre_SWA_resume' and sha(last)==original;checks['ordinary_resume_does_not_rewrite_checkpoint']=True
  missing=save(run,1,51,38250,3.);before=sha(missing)
  try:prepare_resume(1,run);raise RuntimeError('missing archive incorrectly accepted')
  except AssertionError as e:assert 'missing' in str(e)
  assert sha(missing)==before;checks['missing_archive_fails_without_checkpoint_mutation']=True
  save(run,1,48,36000,4.);record=snapshot(1,run);unsafe=save(run,1,52,39000,5.);before=sha(unsafe);archive=Path(record['archive']);archive.write_bytes(archive.read_bytes()+b'corruption')
  try:prepare_resume(1,run);raise RuntimeError('corrupt archive incorrectly accepted')
  except AssertionError as e:assert 'hash mismatch' in str(e)
  assert sha(unsafe)==before;checks['corrupt_archive_fails_without_checkpoint_mutation']=True
  a,b=run/'a',run/'b';a.write_bytes(b'new');b.write_bytes(b'preserve')
  try:copy_verified(a,b);raise RuntimeError('immutable archive overwritten')
  except AssertionError:pass
  assert b.read_bytes()==b'preserve';checks['immutable_destination_is_preserved']=True
 # This check is read-only and must refuse before loading any real checkpoint.
 running=[f for f in [0,1] if active(f,RUN)]
 for f in running:
  try:prepare_resume(f);raise RuntimeError('active trainer recovery was allowed')
  except AssertionError as e:assert 'active' in str(e)
 checks['active_real_trainers_refused_without_checkpoint_read_or_write']=running
 out={'checks':checks,'real_training_interrupted':False,'GPU_used':False};(P/'artifacts/research_audit_20260909/SWA_RECOVERY_TRANSACTION_CHECK.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
