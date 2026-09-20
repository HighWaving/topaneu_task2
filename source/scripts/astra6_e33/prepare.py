"""Source-only raw preprocessing with native-label identity and durable per-case resumption."""
import os,json,pickle,hashlib,time,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from nndet.preprocessing.preprocessor import GenericPreprocessor
P=Path(__file__).resolve().parents[2];RUN=P/'artifacts/astra6_e33_source_only_detector_20260910';TASK='Task133FG_TopAneuMR_TrainOnly';PREP=RUN/'data'/TASK/'preprocessed';E23=P/'artifacts/astra6_e23_MR_oof_candidates_20260909';OLD=P.parent/'nndet_data/Task030FG_TopAneuMR';DATA=P.parent/'data_topaneu26'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');os.replace(t,p)
def case(cid):
 begin=time.monotonic();planfile=PREP/'D3V001_3d.pkl';planhash=sha(planfile);record=RUN/'preprocessing_cases'/f'{cid}.json';dest=RUN/'persistent_preprocessed';dest.mkdir(exist_ok=True)
 if record.exists():
  r=json.loads(record.read_text());assert r['plan_sha256']==planhash
  for name,d in r['files'].items():assert (dest/name).exists() and (dest/name).stat().st_size==d['bytes']
  return r
 identity=json.loads((E23/'evidence/source_identity.json').read_text())[cid];image=DATA/f'images/{cid}_0000.nii.gz';gt=DATA/f'location_masks/{cid}.nii.gz';assert sha(image)==identity['image_sha256'];assert sha(gt)==identity['current_location_mask_sha256']
 corrected=identity.get('rebuilt_from_current_GT',False);label=(E23/'corrected_native_labels' if corrected else OLD/'raw_splitted/labelsTr')/f'{cid}.nii.gz';labelhash=sha(label)
 if not corrected:assert labelhash==identity['detector_instance_label_sha256']
 else:
  audited=json.loads((P/'artifacts/train_only_planning_repair_20260909/PROPERTIES_READY.json').read_text())['updated_native_properties'][cid];assert labelhash==audited['native_label_sha256']
 plan=pickle.loads(planfile.read_bytes());assert cid in plan['dataset_properties']['instance_props_per_patient'];assert plan['dataset_properties']['intensity_properties'] is None
 pre=GenericPreprocessor(norm_scheme_per_modality=plan['normalization_schemes'],use_mask_for_norm=plan['use_mask_for_norm'],transpose_forward=plan['transpose_forward'],intensity_properties=None,resample_anisotropy_threshold=plan['resample_anisotropy_threshold'])
 x,y,props=pre.preprocess_test_case([str(image)],plan['target_spacing'],seg_file=str(label));assert x.shape==y.shape and np.isfinite(x).all();assert y.min()>=-1 and y.max()<32767;props['use_nonzero_mask_for_norm']=pre.use_mask_for_norm;boxes=pre.compute_candidates(data=x,seg=y,properties=props);y=y.astype(np.int16)
 if identity['foreground_voxels']>0:assert np.any(y>0),'All positive source targets disappeared under resampling'
 files={}
 for suffix,item in [('.npy',x),('_seg.npy',y),('.pkl',props),('_boxes.pkl',boxes)]:
  target=dest/(cid+suffix);temp=target.with_suffix(target.suffix+f'.tmp.{os.getpid()}')
  with temp.open('wb') as f:
   if suffix.endswith('npy'):np.save(f,item,allow_pickle=False)
   else:pickle.dump(item,f)
  os.replace(temp,target);files[target.name]={'sha256':sha(target),'bytes':target.stat().st_size}
 r={'case_id':cid,'plan_sha256':planhash,'raw_image_sha256':identity['image_sha256'],'raw_location_GT_sha256':identity['current_location_mask_sha256'],'native_instance_GT_sha256':labelhash,'source_only':True,'shape':list(x.shape),'preprocessed_instances':boxes['instances'],'native_instances':list(props['instances']),'native_positive_voxels':identity['foreground_voxels'],'preprocessed_positive_voxels':int((y>0).sum()),'image_dtype':str(x.dtype),'segmentation_dtype':str(y.dtype),'files':files,'seconds':time.monotonic()-begin};write(record,r);return r

def main():
 if (RUN/'SOURCE_READY.json').exists():return
 started=time.monotonic()
 while not (RUN/'PLAN_READY.json').exists():
  if time.monotonic()-started>3600:raise RuntimeError('Planning not ready after 1h; inspect plan log before retry')
  time.sleep(30)
 planready=json.loads((RUN/'PLAN_READY.json').read_text());plan=pickle.loads((PREP/'D3V001_3d.pkl').read_bytes());cases=sorted(plan['dataset_properties']['instance_props_per_patient']);assert len(cases)==267;assert not set(cases)&set(planready['excluded_comparison_cases']);assert shutil.disk_usage(RUN).free>160*2**30
 dataset=json.loads((OLD/'dataset.json').read_text());dataset.update(name='TopAneuMR_TrainOnly267',task=TASK);write(PREP.parent/'dataset.json',dataset)
 # Same-source patch validation is a stability proxy only; final epoch60 is fixed.
 split=[{'train':cases,'val':cases}];(PREP/'splits_final.pkl').write_bytes(pickle.dumps(split));write(RUN/'source_split.json',{'train':cases,'validation_proxy':cases,'training_validation_overlap_explicit':True,'not_used_for_model_selection':True,'comparison_excluded':planready['excluded_comparison_cases'],'gradient_fit_source267_only':True,'no_claim_of_independent_validation':True})
 completed=[]
 with ProcessPoolExecutor(max_workers=2) as pool:
  futures={pool.submit(case,c):c for c in cases}
  for f in as_completed(futures):
   r=f.result();completed.append(r);write(RUN/'PREPROCESS_PROGRESS.json',{'timestamp':time.time(),'completed':len(completed),'total':len(cases),'recent_case':r['case_id'],'recent_case_seconds':r['seconds'],'workers':2});print('E33 PREPROCESS',len(completed),len(cases),r['case_id'],round(r['seconds'],2),flush=True)
 dest=PREP/'D3V001_3d/imagesTr';dest.mkdir(parents=True,exist_ok=True)
 for r in completed:
  for name,d in r['files'].items():
   target=RUN/'persistent_preprocessed'/name;link=dest/name
   if link.exists():assert link.resolve()==target.resolve()
   else:link.symlink_to(target)
 write(RUN/'SOURCE_READY.json',{'n_cases':267,'plan_sha256':sha(PREP/'D3V001_3d.pkl'),'split_sha256':sha(RUN/'source_split.json'),'seconds':time.monotonic()-started,'files_bytes':sum(v['bytes'] for r in completed for v in r['files'].values()),'per_case_hash_records':'preprocessing_cases','training_inputs':'persistent_preprocessed','no_comparison_preprocessing_before_fit':True});print('E33 SOURCE READY',flush=True)
if __name__=='__main__':main()
