"""Protect an unaveraged checkpoint before legacy, non-serializable SWA.

No changes to a healthy trainer. Recovery preserves the interrupted checkpoint
and replays from a verified pre-SWA checkpoint only after an actual restart.
"""
import argparse,datetime,hashlib,json,os,shutil
from pathlib import Path
P=Path(__file__).resolve().parents[2];RUN=P/'artifacts/astra6_e33_source_only_detector_20260910';TASK='Task133FG_TopAneuMR_TrainOnly';MODEL='RetinaUNetV001_D3V001_3d'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def stamp(p):
 s=Path(p).stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns

def copy_verified(source,dest):
 source,dest=Path(source),Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);before=stamp(source);temp=dest.with_name(dest.name+f'.tmp.{os.getpid()}')
 try:
  shutil.copyfile(source,temp);digest=sha(temp);assert sha(source)==digest and stamp(source)==before,'Source changed during copy; retry at a stable checkpoint'
  if dest.exists():assert sha(dest)==digest,'Refusing to overwrite a different immutable archive';temp.unlink()
  else:os.replace(temp,dest)
  return digest
 finally:
  if temp.exists():temp.unlink()

def read_state(path):
 import torch
 s=torch.load(path,map_location='cpu');return s

def last_path(fold,run=RUN):return Path(run)/'models'/TASK/MODEL/f'fold{fold}'/'model_last.ckpt'
def active(fold,run):
 if Path(run).resolve()!=RUN.resolve():return False # isolated recovery fixtures; production CLI always RUN
 expected=str(P/'scripts/astra6_e33/training_entry.py').encode()
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:argv=(proc/'cmdline').read_bytes().split(b'\0')
  except (OSError,PermissionError):continue
  if expected in argv:
   i=argv.index(expected)
   if len(argv)>i+1 and argv[i+1]==str(fold).encode():return True
 return False

def snapshot(fold,run=RUN):
 run=Path(run);root=run/'checkpoints/pre_swa';lock=root/f'fold{fold}.json';archive=root/f'fold{fold}.ckpt'
 if lock.exists():
  record=json.loads(lock.read_text());assert sha(archive)==record['sha256'];return record
 candidate=archive if archive.exists() else root/f'fold{fold}.candidate.ckpt'
 digest=sha(candidate) if candidate.exists() else copy_verified(last_path(fold,run),candidate)
 try:s=read_state(candidate)
 except Exception:
  if candidate!=archive:candidate.unlink(missing_ok=True)
  raise
 epoch,step=int(s['epoch']),int(s['global_step'])
 if not (36000<=step<=36750 and epoch<=49):
  if candidate!=archive:candidate.unlink()
  raise RuntimeError(f'Need stable pre-SWA checkpoint at36000..36750updates and next epoch<=49; got {step}, {epoch}')
 assert 'optimizer_states' in s and 'lr_schedulers' in s;del s
 if candidate!=archive:os.replace(candidate,archive)
 record={'fold':fold,'global_step':step,'next_epoch':epoch,'sha256':digest,'archive':str(archive),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'Pinned SWA callback does not serialize averaging state. Protect pre-SWA replay point without interrupting healthy training.','normal_full_training_budget':45000};temporary=lock.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');os.replace(temporary,lock);return record

def prepare_resume(fold,run=RUN):
 run=Path(run);assert not active(fold,run),'Refusing checkpoint recovery while this fold trainer is active';last=last_path(fold,run);s=read_state(last);step,epoch=int(s['global_step']),int(s['epoch']);del s
 if epoch<=49 and step<=36750:return {'action':'ordinary_pre_SWA_resume','global_step':step,'next_epoch':epoch}
 root=run/'checkpoints/pre_swa';lock=root/f'fold{fold}.json';archive=root/f'fold{fold}.ckpt'
 assert lock.exists() and archive.exists(),'Unsafe legacy SWA resume: protected pre-SWA checkpoint missing; do not launch trainer blindly'
 record=json.loads(lock.read_text());assert sha(archive)==record['sha256'],'Protected checkpoint hash mismatch';pre=read_state(archive);assert int(pre['epoch'])<=49 and 36000<=int(pre['global_step'])<=36750;del pre
 interrupted_sha=sha(last);backup=run/'checkpoints/interrupted_swa'/f'fold{fold}_step{step}_{interrupted_sha[:16]}.ckpt';assert copy_verified(last,backup)==interrupted_sha
 staged=last.with_name(f'model_last.pre_swa_recovery.{os.getpid()}.ckpt');assert copy_verified(archive,staged)==record['sha256'];assert not active(fold,run);assert sha(last)==interrupted_sha,'Current checkpoint changed during recovery; refusing replacement';os.replace(staged,last)
 result={'action':'replay_from_protected_pre_SWA','interrupted_step':step,'interrupted_next_epoch':epoch,'interrupted_backup':str(backup),'interrupted_sha256':interrupted_sha,'restored_step':record['global_step'],'restored_sha256':record['sha256'],'planned_total_updates':45000,'previous_work_replayed_updates':max(0,step-record['global_step']),'worker_RNG_bitwise_replay_not_claimed':True,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()};log=run/'checkpoints/SWA_RECOVERY.jsonl'
 with log.open('a') as f:f.write(json.dumps(result)+'\n')
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--snapshot',action='store_true');p.add_argument('--fold',type=int,choices=[0,1],required=True);a=p.parse_args();assert a.snapshot,'Recovery is called by train_fold only on an actual resume';print(json.dumps(snapshot(a.fold),indent=2),flush=True)
if __name__=='__main__':main()
