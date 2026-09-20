import json,time,re
from pathlib import Path
import numpy as np
from scripts.astra6_e12.common import *
from scripts.astra6_e01.e01_common import load_boxes,box_to_native_bounds,load_nifti,write_json,sha256_file
from scripts.astra6_e03.run_e03 import component_records_fast

def group(cid):return re.sub(r'(_ct_\d+)_\d+$',r'\1',cid)
def prepare():
 for s in ['features/cases','model','logs','evaluation']:(RUN/s).mkdir(parents=True,exist_ok=True)
 files=sorted((P/'artifacts/fold1_eval_center1_epoch60').glob('*_boxes.pkl'))+sorted((P/'artifacts/fold1_source_center5_epoch60_20260909').glob('*_boxes.pkl'))
 files=[f for f in files if (DATA/f'images/{f.name.removesuffix("_boxes.pkl")}_0000.nii.gz').exists()]
 ids=[f.name.removesuffix('_boxes.pkl') for f in files];assert len(ids)==267 and not any('center2' in c or '_ct_' in c for c in ids)
 parent=P/'artifacts/astra6_e06_image_fp_filter_20260909';oldsplit=json.loads((parent/'source_split.json').read_text());dev={group(c) for c in oldsplit['development_cases'] if '_mr_' in c};new=sorted(c for c in ids if 'center5' in c);np.random.default_rng(SEED).shuffle(new);dev.update(new[:round(.2*len(new))])
 write_json(RUN/'source_split.json',{'train_cases':[c for c in ids if group(c) not in dev],'development_cases':[c for c in ids if group(c) in dev],'source_detector_in_sample':True,'center2_used_for_fit':False,'parent_MR_split_preserved':True,'new_center5_source_cases':68})
 for cid in ids:
  if 'center1' not in cid:continue
  meta=parent/f'features/cases/{cid}.json';rr=json.loads(meta.read_text());assert all(r['development']==(group(cid) in dev) for r in rr)
  for suffix in ['.npz','.json','.geometry']:
   src=parent/f'features/cases/{cid}{suffix}';dest=RUN/f'features/cases/{cid}{suffix}'
   if src.exists() and not dest.exists():dest.symlink_to(src)
 config={'experiment':'E12','hypothesis':'68 additional center5 source MR cases increase lesion appearance diversity and improve source-calibrated MR-only FP rejection beyond E08D, compared with current E06 best','control':'E06 MR filter + E02 C + E04 S; no ensemble','change':'same image-only two-scale architecture;267MR source cases instead of199MR, preserve prior199source case split and add20percent center5 dev','source_detector_in_sample':True,'candidate_pool':{'min_score':.05,'topk':20},'training':{'max_epochs':100,'min_stop_epoch':30,'patience':20,'batch':32,'AdamW_lr':.0003,'weight_decay':.01,'final_epochs':'source-selected duration','budget_hours':4,'checkpoint':'everyepoch optimizer RNG sampler RNG'},'threshold':'minimum source-dev positive probability in original score.3/top5 deployment pool; fixed before MR40 inference','adoption':'MR40 all-six Pareto gain versus E06 and no loss of matched lesions; otherwise retain E06','no_center2_or_CT_fit':True}
 write_json(RUN/'config.json',config);records=[];start=time.time()
 for n,(cid,f) in enumerate(zip(ids,files),1):
  dest=RUN/f'features/cases/{cid}.npz';meta=dest.with_suffix('.json')
  if dest.exists() and meta.exists():records+=json.loads(meta.read_text());continue
  arr,aff,norm=load_image(cid);gt,ga,gs=load_nifti(DATA/f'location_masks/{cid}.nii.gz');assert arr.shape==gt.shape
  corners=np.array(np.meshgrid(*[(0,d-1) for d in arr.shape],indexing='ij')).reshape(3,-1).T;delta=corners@(aff[:3,:3]-ga[:3,:3]).T+(aff[:3,3]-ga[:3,3]);geometry_error_mm=float(np.linalg.norm(delta,axis=1).max());identity,voxel_bound=nearest_label_identity(aff,ga,arr.shape);assert identity,(cid,geometry_error_mm,voxel_bound)
  write_json(RUN/f'features/cases/{cid}.geometry',{'max_image_GT_corner_error_mm':geometry_error_mm,'nearest_neighbor_physical_resampling_is_identity':identity,'max_voxel_index_error_per_axis':voxel_bound,'reason':'All voxel centers map to the same nearest label indices, proven by affine corner bounds; source arrays and original files unchanged'})
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
