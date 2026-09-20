"""Resume own two folds from first complete epoch with faster augmentation."""
import os,sys,time,json,signal,hashlib,shutil,subprocess
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import torch
P=Path(__file__).resolve().parents[2];R=P/'artifacts/astra6_e23_MR_oof_candidates_20260909';OLD={0:1122856,1:1122863}
def descendants(root):
 parents={}
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   try:parents[int(p.name)]=int((p/'stat').read_text().rsplit(')',1)[1].split()[1])
   except (OSError,ValueError):pass
 found={root}
 while True:
  new={pid for pid,ppid in parents.items() if ppid in found}-found
  if not new:return sorted(found)
  found|=new

def main():
 pending={0,1};running={};logs=[]
 while pending or running:
  for fold in list(pending):
   d=R/f'models/Task130FG_TopAneuMR_OOF/RetinaUNetV001_D3V001_3d/fold{fold}';checkpoint=d/'model_last.ckpt';mark=R/f'logs/fold{fold}_WORKER_RESUME.json'
   if mark.exists():pending.remove(fold);continue
   if not checkpoint.exists():continue
   stat=checkpoint.stat();time.sleep(2)
   if (stat.st_size,stat.st_mtime_ns)!=(checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns):continue
   try:state=torch.load(checkpoint,map_location='cpu')
   except Exception:continue
   epoch=int(state['epoch']);step=int(state['global_step']);assert epoch>=0
   backup=R/f'checkpoints/fold{fold}_before_worker_scaling.ckpt';shutil.copy2(checkpoint,backup);digest=hashlib.sha256(backup.read_bytes()).hexdigest();del state
   pid=OLD[fold];cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode();assert 'scripts.astra6_e23.train_fold' in cmd and f'--fold {fold}' in cmd,(pid,cmd);tree=descendants(pid)
   for q in tree:
    try:os.kill(q,signal.SIGTERM)
    except ProcessLookupError:pass
   # Do not launch until own old processes have released their GPU context.
   for _ in range(30):
    live=[]
    for q in tree:
     f=Path(f'/proc/{q}/stat')
     if f.exists():
      try:
       if f.read_text().rsplit(')',1)[1].split()[0]!='Z':live.append(q)
      except OSError:pass
    if not live:break
    time.sleep(1)
   assert not live,('Own old process not terminated; refusing GPU overlap',live)
   assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
   log=(R/f'logs/fold{fold}_workflow_resumed.log').open('a');logs.append(log);proc=subprocess.Popen([sys.executable,'-m','scripts.astra6_e23.train_fold','--fold',str(fold)],cwd=P,stdout=log,stderr=subprocess.STDOUT);running[fold]=proc;pending.remove(fold);mark.write_text(json.dumps({'previous_process_tree':tree,'resume_driver_pid':proc.pid,'checkpoint_epoch_zero_based':epoch,'checkpoint_global_step':step,'backup':str(backup),'sha256':digest,'old_workers':3,'new_workers':7,'budget_unchanged':True,'time':time.time()},indent=2)+'\n');print('E23_RESUMED_WITH_7_WORKERS',fold,epoch,step,flush=True)
  for fold,proc in list(running.items()):
   code=proc.poll()
   if code is not None:
    print('E23_RESUMED_DRIVER_FINISHED',fold,code,flush=True);del running[fold]
    if code:raise RuntimeError(f'Fold{fold} failed; inspect resumed log')
  if pending or running:time.sleep(10)
if __name__=='__main__':main()
