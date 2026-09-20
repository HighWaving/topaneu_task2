"""Automatically complete source shape helper and all held-out detector inference."""
import argparse,json,os,subprocess,time,sys,hashlib
from pathlib import Path
P=Path(__file__).resolve().parents[2];V=P.parent;RUN=P/'artifacts/astra6_e23_MR_oof_candidates_20260909';TASK='Task130FG_TopAneuMR_OOF';MODEL='RetinaUNetV001_D3V001_3d';GPUS=['GPU-a643ded3-193b-58e8-b362-be93dc8eac14','GPU-f7491bbb-3972-1264-755a-63dec96b4a7d']
def main():
 from scripts.astra6_e23.local_cache import restore_missing
 restore_missing()
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=[0,1],required=True);a=ap.parse_args();ready=RUN/f'checkpoints/fold{a.fold}_TRAINED.json';start=time.monotonic()
 while not ready.exists():
  assert time.monotonic()-start<36*3600,'Source detector exceeded36hour workflow budget';time.sleep(10)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPUS[a.fold],OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1');py=V/'conda_envs/nnunet_v100/bin/python';subprocess.run([str(py),'-m','scripts.astra6_e23.train_segmenter_fold','--fold',str(a.fold)],cwd=P,env=env,check=True)
 split=json.loads((RUN/'source_split.json').read_text())['folds'][a.fold];ids=split['val'];assert not set(ids)&set(split['train']);fold=RUN/'models'/TASK/MODEL/f'fold{a.fold}';lock=json.loads(ready.read_text());assert hashlib.sha256((fold/'model_last.ckpt').read_bytes()).hexdigest()==lock['sha256'];app=V/'new_aneurysms';env.update(PYTHONPATH=os.pathsep.join([str(V/'external/nnDetection'),str(app)]),det_data=str(RUN/'data'),det_models=str(RUN/'models'),det_num_threads='2');dest=RUN/f'oof_boxes/fold{a.fold}';dest.mkdir(parents=True,exist_ok=True)
 cmd=[str(V/'conda_envs/nndet/bin/python'),str(app/'src/scripts/infer_nndet_adam.py'),'--training-dir',str(fold),'--source-dir',str(RUN/'data'/TASK/'preprocessed/D3V001_3d/imagesTr'),'--output-dir',str(dest),'--checkpoint','last','--device','cuda:0','--num-tta','1','--batch-size','1','--max-detections','200']
 for cid in ids:cmd.extend(['--case-id',cid])
 (RUN/f'logs/fold{a.fold}_INFERENCE_COMMAND.json').write_text(json.dumps({'command':cmd,'heldout_only':True},indent=2)+'\n');subprocess.run(cmd,cwd=app,env=env,check=True)
 hashes={}
 for cid in ids:
  f=dest/f'{cid}_boxes.pkl';assert f.exists();hashes[cid]=hashlib.sha256(f.read_bytes()).hexdigest()
 (RUN/f'checkpoints/fold{a.fold}_OOF_COMPLETE.json').write_text(json.dumps({'fold':a.fold,'cases':ids,'checkpoint_sha256':lock['sha256'],'boxes_sha256':hashes,'heldout_from_detector_and_segmenter':True,'no_GT_vessel_input':True,'source_split_sha256':hashlib.sha256((RUN/'source_split.json').read_bytes()).hexdigest()},indent=2)+'\n');print('E23_FOLD_OOF_COMPLETE',a.fold,len(ids),flush=True)
if __name__=='__main__':main()
