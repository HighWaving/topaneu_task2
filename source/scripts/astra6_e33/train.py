"""One 45,000-update deployment candidate; do not re-run E23 OOF models."""
import json,os,subprocess,time
from pathlib import Path
from scripts.astra6_e33.prepare import P,RUN,PREP,TASK,sha,write
MODEL='RetinaUNetV001_D3V001_3d'
def main():
 assert (RUN/'SOURCE_READY.json').exists();out=RUN/'models'/TASK/MODEL/'fold0';ready=RUN/'TRAINED.json'
 if ready.exists():return
 from scripts.astra6_e33.cache import restore_missing
 restore_missing();(RUN/'logs').mkdir(exist_ok=True);(RUN/'checkpoints').mkdir(exist_ok=True);mode='resume' if (out/'model_last.ckpt').exists() else 'overwrite'
 if mode=='resume':
  from scripts.astra6_e33.safe_swa_resume import prepare_resume
  print('E33_RESUME',json.dumps(prepare_resume(0)),flush=True)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-f7491bbb-3972-1264-755a-63dec96b4a7d',det_data=str(RUN/'data'),det_models=str(RUN/'models'),det_num_threads='8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',det_verbose='0',PYTHONUNBUFFERED='1',PYTHONPATH=str(P.parent/'external/nnDetection'));py=P.parent/'conda_envs/nndet/bin/python';overrides=['exp.fold=0',f'train.mode={mode}','trainer_cfg.max_num_epochs=50','trainer_cfg.swa_epochs=10','trainer_cfg.num_train_batches_per_epoch=750','trainer_cfg.num_val_batches_per_epoch=100','augment_cfg.num_cached_per_thread=1'];cmd=[str(py),str(P/'scripts/astra6_e33/training_entry.py'),'0',TASK,*overrides];write(RUN/'logs/TRAIN_COMMAND.json',{'command':cmd,'mode':mode,'source_only':True,'seed':20260910,'GPU':3,'augmentation_worker_limit':7,'full_budget':45000});start=time.monotonic();subprocess.run(cmd,cwd=P,env=env,check=True)
 import torch
 state=torch.load(out/'model_last.ckpt',map_location='cpu');assert int(state['global_step'])==45000 and int(state['epoch'])>=59;assert state['E33_source_only_fit']['source_ready']['plan_sha256']==sha(PREP/'D3V001_3d.pkl');write(ready,{'epoch':int(state['epoch']),'global_step':int(state['global_step']),'checkpoint':str(out/'model_last.ckpt'),'sha256':sha(out/'model_last.ckpt'),'plan_sha256':sha(PREP/'D3V001_3d.pkl'),'source_cases':267,'no_MR40_gradient_or_supervised_planning_input':True,'repeated_MR40_development_limitation_remains':True,'seconds_this_execution':time.monotonic()-start});print('E33 FULL FORMAL TRAINING COMPLETE',flush=True)
if __name__=='__main__':main()
