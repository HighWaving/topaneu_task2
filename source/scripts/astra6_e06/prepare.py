import json,time,re
from pathlib import Path
import numpy as np
from scripts.astra6_e06.common import *
from scripts.astra6_e01.e01_common import load_boxes,box_to_native_bounds,load_nifti,write_json,sha256_file
from scripts.astra6_e03.run_e03 import component_records_fast

def group(cid):return re.sub(r'(_ct_\d+)_\d+$',r'\1',cid)
def prepare():
 for s in ['features/cases','model','logs','evaluation']:(RUN/s).mkdir(parents=True,exist_ok=True)
 files=sorted((P/'artifacts/fold1_eval_center1_epoch60').glob('*_boxes.pkl'))+sorted((P/'artifacts/ct_fold2_all109_boxes').glob('*center4*_boxes.pkl'))
 files=[f for f in files if (DATA/f'images/{f.name.removesuffix("_boxes.pkl")}_0000.nii.gz').exists()]
 ids=[f.name.removesuffix('_boxes.pkl') for f in files];assert not any('center2' in c for c in ids)
 rng=np.random.default_rng(SEED);dev=set()
 for mod in ['mr','ct']:
  groups=sorted({group(c) for c in ids if f'_{mod}_' in c});rng.shuffle(groups);dev.update(groups[:round(.2*len(groups))])
 write_json(RUN/'source_split.json',{'train_cases':[c for c in ids if group(c) not in dev],'development_cases':[c for c in ids if group(c) in dev],'group_rule':'CT scan suffixes grouped','source_detector_in_sample':True,'center2_used_for_fit':False})
 config={'experiment':'E06','hypothesis':'two-scale raw-image classifier rejects detector false positives while preserving lesion recall','source':'existing epoch60 MR center1 and frozen CT center4 predicted boxes; official aneurysm GT for binary training labels only','source_detector_in_sample':True,'limitation':'source validation optimistic for upstream detector; fixed MR40 not used for fitting/calibration','candidate_pool':{'min_score':.05,'topk':20},'positive':'candidate box contains at least10% of a GT component; small partial overlap ignored','negative':'zero GT voxels within candidate box','inputs':['box-normalized image32cube','physical RAS image32cube at1mm'],'no_vessel_or_detector_score_input':True,'model':'shared3D CNN16/32/64 over two scales, binary head','training':{'max_epochs':100,'min_epochs':30,'patience':20,'batch':32,'AdamW_lr':.0003,'weight_decay':.01,'selection':'source development balanced BCE','final_epochs':'source-selected duration','budget_hours':6,'checkpoint':'everyepoch optimizer RNG sampler RNG','augmentation':'shared gamma .8..1.25, Gaussian noise .01, axis flips'},'threshold':'lowest source-development original-positive probability, nextafter toward0, preserving all observed source positives; fixed before MR40','gate':'MR40 paired official no degradation and lowerFP; preserve E04 otherwise'}
 write_json(RUN/'config.json',config);records=[];start=time.time()
 for n,(cid,f) in enumerate(zip(ids,files),1):
  dest=RUN/f'features/cases/{cid}.npz';meta=dest.with_suffix('.json')
  if dest.exists() and meta.exists():records+=json.loads(meta.read_text());continue
  arr,aff,norm=load_image(cid);gt,ga,gs=load_nifti(DATA/f'location_masks/{cid}.nii.gz');assert arr.shape==gt.shape
  corners=np.array(np.meshgrid(*[(0,d-1) for d in arr.shape],indexing='ij')).reshape(3,-1).T;delta=corners@(aff[:3,:3]-ga[:3,:3]).T+(aff[:3,3]-ga[:3,3]);geometry_error_mm=float(np.linalg.norm(delta,axis=1).max());assert geometry_error_mm<=.001,(cid,geometry_error_mm)
  write_json(RUN/f'features/cases/{cid}.geometry',{'max_image_GT_corner_error_mm':geometry_error_mm,'allowed_mm':.001,'reason':'NIfTI header rounding only; no label resampling or original-file edits'})
  comps=component_records_fast(gt);bx,sc,_=load_boxes(f);case=[];xx=[];crng=np.random.default_rng(SEED+sum(map(ord,cid)))
  def target(lo,hi):
   counts=[int(np.all((c['coords']>=lo)&(c['coords']<hi),axis=1).sum()) for c in comps]
   if any(cnt/max(1,len(c['coords']))>=.1 for cnt,c in zip(counts,comps)):return 1
   return -1 if sum(counts)>0 else 0
  candidates=sorted([(i,float(s)) for i,s in enumerate(sc) if s>=.05],key=lambda z:(-z[1],z[0]))[:20]
  for i,score in candidates:
   lo,hi=box_to_native_bounds(bx[i]);y=target(lo,hi)
   if y<0:continue
   bounds=[(lo,hi)]
   if y and group(cid) not in dev:
    for _ in range(3):
     extent=hi-lo;shift=crng.uniform(-.1,.1,3)*extent;scale=crng.uniform(.9,1.1);center=(hi+lo)/2+shift;l=center-extent*scale/2;h=center+extent*scale/2
     if target(l,h)==1:bounds.append((l,h))
   for aug,(l,h) in enumerate(bounds):
    xx.append(crops(arr,aff,l,h).astype(np.float16));case.append({'case_id':cid,'group':group(cid),'development':group(cid) in dev,'original_index':i,'augmentation':aug,'score':score,'low':l.tolist(),'high':h.tolist(),'y':y,'case_row':len(xx)-1})
  np.savez_compressed(dest,x=np.stack(xx) if xx else np.empty((0,2,32,32,32),np.float16));write_json(meta,case);records+=case
  write_json(RUN/'features/progress.json',{'cases':n,'total_cases':len(ids),'rows':len(records),'seconds':time.time()-start});print(cid,n,len(ids),'rows',len(case),'positive',sum(r['y'] for r in case),flush=True)
 x=np.lib.format.open_memmap(RUN/'features/images.npy',mode='w+',dtype=np.float16,shape=(len(records),2,32,32,32));offset=0
 for cid in ids:
  a=np.load(RUN/f'features/cases/{cid}.npz')['x'];x[offset:offset+len(a)]=a;offset+=len(a)
 x.flush();assert offset==len(records) and np.isfinite(x).all();(RUN/'features/records.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n')
 write_json(RUN/'features/READY.json',{'cases':len(ids),'rows':len(records),'positive':sum(r['y'] for r in records),'images_sha256':sha256_file(RUN/'features/images.npy'),'records_sha256':sha256_file(RUN/'features/records.jsonl'),'box_hashes':{f.name:sha256_file(f) for f in files}});print('SOURCE_READY',flush=True)
if __name__=='__main__':prepare()
