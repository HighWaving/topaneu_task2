"""Run an isolated, from-scratch nnDetection OOF fold with epoch-resume support."""
import argparse,json,os,sys,time,subprocess
from pathlib import Path
# Keep this driver compatible with the pinned Py3.9 detector environment.
P=Path(__file__).resolve().parents[2];RUN=P/'artifacts/astra6_e23_MR_oof_candidates_20260909';V=P.parent;TASK='Task130FG_TopAneuMR_OOF';MODEL='RetinaUNetV001_D3V001_3d';GPUS=['GPU-a643ded3-193b-58e8-b362-be93dc8eac14','GPU-f7491bbb-3972-1264-755a-63dec96b4a7d']
def main():
 from scripts.astra6_e23.local_cache import restore_missing
 restore_missing()
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=[0,1],required=True);a=ap.parse_args();start=time.monotonic()
 while not (RUN/'SOURCE_READY.json').exists():
  assert time.monotonic()-start<21600,'Source integrity preparation exceeded six hours';time.sleep(10)
 config=json.loads((RUN/'config.json').read_text());fold=RUN/'models'/TASK/MODEL/f'fold{a.fold}';ready=RUN/f'checkpoints/fold{a.fold}_TRAINED.json'
 if ready.exists():return
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[a.fold],det_data=str(RUN/'data'),det_models=str(RUN/'models'),det_num_threads='8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',det_verbose='0',PYTHONUNBUFFERED='1',PYTHONPATH=str(V/'external/nnDetection'))
 mode='resume' if (fold/'model_last.ckpt').exists() else 'overwrite'
 if mode=='resume':
  from scripts.astra6_e23.safe_swa_resume import prepare_resume
  print('E23_RESUME_SAFETY',json.dumps(prepare_resume(a.fold)),flush=True)
 overrides=[f'exp.fold={a.fold}',f'train.mode={mode}','trainer_cfg.max_num_epochs=50','trainer_cfg.swa_epochs=10','trainer_cfg.num_train_batches_per_epoch=750','trainer_cfg.num_val_batches_per_epoch=100','augment_cfg.num_cached_per_thread=1']
 cmd=[sys.executable,str(P/'scripts/astra6_e23/training_entry.py'),str(a.fold),TASK,*overrides];(RUN/f'logs/fold{a.fold}_COMMAND.json').write_text(json.dumps({'command':cmd,'gpu':GPUS[a.fold],'initialization':'from scratch or same-run resume only','mode':mode},indent=2)+'\n');subprocess.run(cmd,cwd=P,env=env,check=True)
 import torch,hashlib
 state=torch.load(fold/'model_last.ckpt',map_location='cpu');epoch=int(state['epoch']);assert epoch>=59,(epoch,'incomplete60epochs');h=hashlib.sha256((fold/'model_last.ckpt').read_bytes()).hexdigest();ready.write_text(json.dumps({'fold':a.fold,'completed_epoch_zero_based':epoch,'global_step':int(state['global_step']),'checkpoint':str(fold/'model_last.ckpt'),'sha256':h,'no_comparison_case_fit':True,'elapsed_seconds':time.monotonic()-start},indent=2)+'\n');print('E23_FOLD_FULL_TRAINING_COMPLETE',a.fold,epoch,flush=True)
if __name__=='__main__':main()
