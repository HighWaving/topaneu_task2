"""Persist actual scalar progress and allocated-GPU resource observations."""
from pathlib import Path
import json,time,datetime,subprocess,os
P=Path(__file__).resolve().parents[2];R=P/'artifacts/astra6_e23_MR_oof_candidates_20260909'
def snapshot():
 out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'folds':{}}
 for fold in [0,1]:
  d=R/f'models/Task130FG_TopAneuMR_OOF/RetinaUNetV001_D3V001_3d/fold{fold}';scalars={}
  for md in d.glob('mlruns/*/*/metrics'):
   for f in md.rglob('*'):
    if f.is_file():
     lines=f.read_text().splitlines()
     if lines:
      a=lines[-1].split()
      if len(a)==3:
       key=str(f.relative_to(md));row={'timestamp_ms':int(a[0]),'value':float(a[1]),'step':int(a[2])}
       if key not in scalars or row['timestamp_ms']>scalars[key]['timestamp_ms']:scalars[key]=row
  out['folds'][str(fold)]={'scalars':scalars,'full_training_complete':(R/f'checkpoints/fold{fold}_TRAINED.json').exists(),'OOF_complete':(R/f'checkpoints/fold{fold}_OOF_COMPLETE.json').exists(),'checkpoint_exists':(d/'model_last.ckpt').exists()}
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);out['allocated_GPU']=[s.strip() for s in gpu.splitlines() if s.startswith(('2,','3,'))]
 dest=R/'LIVE_PROGRESS.json';tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');os.replace(tmp,dest)
 with (R/'logs/resource_progress.jsonl').open('a') as f:f.write(json.dumps(out)+'\n')
 return out
if __name__=='__main__':
 while True:
  s=snapshot()
  if all(f['OOF_complete'] for f in s['folds'].values()):break
  time.sleep(30)
