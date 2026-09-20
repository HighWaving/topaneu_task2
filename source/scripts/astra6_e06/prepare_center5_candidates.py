"""Extend source-only FP training candidates using frozen detector and existing image preprocessing."""
import json,os,subprocess
from pathlib import Path
from scripts.astra6_e01.e01_common import P,V,DATA,sha256_file,write_json
out=P/'artifacts/fold1_source_center5_epoch60_20260909';out.mkdir(exist_ok=True);ids=sorted(p.name.removesuffix('_0000.nii.gz') for p in (DATA/'images').glob('*center5_mr*_0000.nii.gz'));assert len(ids)==68 and not any('center2' in c for c in ids);source=V/'nndet_data/Task030FG_TopAneuMR/preprocessed/D3V001_3d/imagesTr';assert all((source/f'{c}.npz').exists() for c in ids)
checkpoint=P/'artifacts/fold1_checkpoint_snapshots/epoch60';assert sha256_file(checkpoint/'model_last.ckpt')=='f84488d9dfce235fc993c5dc9faca4fd35501b134b1a26a0248b4a309443980e';env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-a643ded3-193b-58e8-b362-be93dc8eac14',OMP_NUM_THREADS='4',PYTHONPATH=str(V/'new_aneurysms'),det_data=str(V/'nndet_data'),det_models=str(V/'nndet_models'))
write_json(out/'PURPOSE.json',{'purpose':'prepare source-domain diversity for subsequent evidence-driven FP training; no new training or holdout fitting in this step','case_ids':ids,'detector_source_in_sample':True,'loaded_input':'nnDetection predict_dir reads only npz data (1image channel); seg is not loaded; original image geometry used for restoration','detector_sha256':sha256_file(checkpoint/'model_last.ckpt'),'deployment_batch_size':1})
cmd=[str(V/'conda_envs/nndet/bin/python'),str(V/'new_aneurysms/src/scripts/infer_nndet_adam.py'),'--training-dir',str(checkpoint),'--source-dir',str(source),'--output-dir',str(out),'--checkpoint','last','--device','cuda:0','--num-tta','1','--batch-size','1','--max-detections','200']
for c in ids:cmd+=['--case-id',c]
subprocess.run(cmd,cwd=V/'new_aneurysms',env=env,check=True);files=sorted(out.glob('*_boxes.pkl'));assert len(files)==68;write_json(out/'READY.json',{'n_cases':68,'boxes_sha256':{p.name:sha256_file(p) for p in files}})
