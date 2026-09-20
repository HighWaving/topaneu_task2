from pathlib import Path
import json,time,shutil
import numpy as np,torch
from scripts.astra6_e01.e01_common import P,DATA,write_json,sha256_file,load_nifti_geometry
from scripts.astra6_e04 import run_e04 as e04
from scripts.delivery.geometry import vessel_geometry_fast
from scripts.astra6_e10.features import contact_features,mirror_features,mask_from_probability,geometry_test,FIELDS,FEATURE_VERSION
BASE=P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z';SHAPE=P/'artifacts/astra6_e04_crop_segmentation_20260909T031146Z';RUN=P/'artifacts/astra6_e10_mask_contact_location_20260909';SPLIT=P/'artifacts/astra6_e05_learned_location_splits_20260909/source_split.json'

def predict_masks(weights,dest):
 if dest.exists() and dest.with_suffix('.json').exists():
  assert json.loads(dest.with_suffix('.json').read_text())['weights_sha256']==sha256_file(weights);return
 model=e04.Segmenter().cuda();model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=False)['state_dict']);model.eval();images=np.load(SHAPE/'features/images.npy',mmap_mode='r');out=np.lib.format.open_memmap(dest,mode='w+',dtype=np.uint8,shape=images.shape)
 with torch.inference_mode():
  for start in range(0,len(images),64):
   a=np.asarray(images[start:start+64],np.float32);x=np.stack([a,np.broadcast_to(e04.PRIOR,a.shape)],axis=1);prob=model(torch.from_numpy(x).cuda()).sigmoid().cpu().numpy()[:,0]
   for j,p in enumerate(prob):out[start+j]=mask_from_probability(p)
 out.flush();write_json(dest.with_suffix('.json'),{'weights_sha256':sha256_file(weights),'masks_sha256':sha256_file(dest),'rows':len(out),'GT_targets_not_read_for_prediction':True});del model;torch.cuda.empty_cache();print('predicted_source_masks',dest.name,flush=True)

def main():
 torch.set_num_threads(4)
 for s in ['model','features/cases','logs','evaluation','source_segmenter/model']:(RUN/s).mkdir(parents=True,exist_ok=True)
 write_json(RUN/'GEOMETRY_TEST.json',geometry_test());rec=[json.loads(s) for s in (BASE/'features/train_records.jsonl').read_text().splitlines()];split=json.loads(SPLIT.read_text());dev=set(split['development_cases']);tr=[i for i,r in enumerate(rec) if r['case_id'] not in dev];assert len(tr)==3978 and not any('center2' in r['case_id'] for r in rec);write_json(RUN/'source_split.json',split)
 config={'experiment':'E10','hypothesis':'predicted tumor-surface contacts with vessel labels improve fine-branch location discrimination beyond candidate-center distances','new_features':396,'total_features':1339,'shape':'frozenE04 for final features and runtime; same output masks','source_validation_shape':'new helper identicalS trained13epochs on source classifier training groups only; no source-dev cases in its fitting','classifier':'sameE02 ExtraTrees512/depth16/leaf2/max_features.5/seed20260905; no architecture search','budget_hours':3,'source':'existing218cases4986rows, no center2','source_vessels':'official provided organizer-predicted silver; runtimeTA36 only','surface_sampling':'max256 deterministic surface points; no GT aneurysm masks as runtime features','gate':'MR40 all-six Pareto improvement; retain currentbest otherwise'};write_json(RUN/'config.json',config)
 helper=RUN/'source_segmenter';link=helper/'features'
 if not link.exists():link.symlink_to(SHAPE/'features',target_is_directory=True)
 if not (helper/'model/LOCKED.json').exists():
  model=e04.fit(helper,'final',tr,epochs=13);write_json(helper/'model/LOCKED.json',{'model_sha256':sha256_file(helper/'model/final_last.pt'),'epochs':13,'train_rows':tr,'excluded_source_development_cases':sorted(dev),'purpose':'source-validation feature generator only, not deployed'});del model;torch.cuda.empty_cache()
 predict_masks(helper/'model/final_last.pt',RUN/'features/development_masks.npy');predict_masks(SHAPE/'model/final_last.pt',RUN/'features/final_masks.npy')
 masks={k:np.load(RUN/f'features/{k}_masks.npy',mmap_mode='r') for k in ['development','final']};rowmap=np.load(SHAPE/'features/rows.npz')['crop_index'];schema=json.loads((BASE/'feature_schema.json').read_text());pair={int(k):int(v) for k,v in schema['vessel_lr_pair'].items()};bycase={}
 for i,r in enumerate(rec):bycase.setdefault(r['case_id'],{}).setdefault(int(rowmap[i]),[]).append(i)
 for n,(cid,groups) in enumerate(sorted(bycase.items()),1):
  cache=RUN/f'features/cases/{cid}.npz'
  if cache.exists():continue
  aff,shape=load_nifti_geometry(DATA/f'images/{cid}_0000.nii.gz');geom=vessel_geometry_fast(DATA/f'vessel_masks/{cid}.nii.gz',shape,aff);indices=[];extras={k:[] for k in masks}
  for crop,ix in groups.items():
   assert len(ix)==2;r=rec[ix[0]];lo=np.array(r['low']);hi=np.array(r['high']);f={k:contact_features(geom,aff,lo,hi,a[crop].astype(bool)) for k,a in masks.items()}
   for i in ix:
    indices.append(i)
    for k in masks:extras[k].append(mirror_features(f[k],pair) if rec[i]['view']=='mirror' else f[k])
  np.savez_compressed(cache,indices=indices,**{k:np.asarray(v,np.float32) for k,v in extras.items()});print('contact_features',n,len(bycase),cid,flush=True)
 old=np.load(BASE/'features/train.npz');extra={k:np.empty((len(rec),396),np.float32) for k in masks}
 for cid in bycase:
  data=np.load(RUN/f'features/cases/{cid}.npz')
  for k in masks:extra[k][data['indices']]=data[k]
 for k in masks:
  X=np.concatenate([old['X'],extra[k]],axis=1);assert X.shape==(4986,1339) and np.isfinite(X).all();np.savez_compressed(RUN/f'features/{k}.npz',X=X,y=old['y'])
 shutil.copy2(BASE/'features/train_records.jsonl',RUN/'features/train_records.jsonl');write_json(RUN/'feature_schema.json',{'feature_version':FEATURE_VERSION,'n_features':1339,'base_schema':str(BASE/'feature_schema.json'),'extra_features':[f'v{v:02d}_{field}' for v in range(1,37) for field in FIELDS],'mirror_offset_x_index':5,'mirror_relative_x_index':8});write_json(RUN/'features/READY.json',{'development_sha256':sha256_file(RUN/'features/development.npz'),'final_sha256':sha256_file(RUN/'features/final.npz'),'records_sha256':sha256_file(RUN/'features/train_records.jsonl'),'helper_segmentation_sha256':sha256_file(helper/'model/final_last.pt'),'runtime_segmentation_sha256':sha256_file(SHAPE/'model/final_last.pt'),'source_cases':218});print('SOURCE_READY',flush=True)
if __name__=='__main__':main()
