"""Source-only audit: does the MR segmenter lose accuracy on actual detector crops?"""
from pathlib import Path
import json,shutil
import numpy as np,torch
from scipy.ndimage import map_coordinates
from torch.utils.data import DataLoader
from scripts.astra6_e01.e01_common import P,DATA,load_boxes,select_candidates,load_nifti,load_nifti_geometry,write_json,sha256_file
from scripts.astra6_e04.run_e04 import Segmenter,SegData,SUPPORT,PRIOR,largest,crop_coordinates,component_records_fast
from scripts.astra6_e12.common import nearest_label_identity
RUN=P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909';PARENT=P/'artifacts/astra6_e04_crop_segmentation_20260909T031146Z';F=P/'artifacts/astra6_e12_MR_expanded_fp_filter_20260909';HELPER=P/'artifacts/astra6_e10_mask_contact_location_20260909/source_segmenter/model/final_last.pt'
def evaluate(weights,indices):
 model=Segmenter().cuda();model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=False)['state_dict']);model.eval();scores=[]
 with torch.inference_mode():
  for x,y in DataLoader(SegData(RUN,indices),batch_size=32,num_workers=0):
   pp=model(x.cuda()).sigmoid().cpu().numpy()[:,0]*SUPPORT
   for prob,gt in zip(pp,y.numpy()[:,0]>0):
    mask=largest(prob>=.5)
    if not mask.any():mask=PRIOR>0
    scores.append(float(2*(mask&gt).sum()/max(1,mask.sum()+gt.sum())))
 del model;torch.cuda.empty_cache();return {'n':len(scores),'crop_Dice_mean':float(np.mean(scores)),'scores':scores}

def main():
 torch.set_num_threads(4)
 for s in ['features/cases','model','evaluation','predictions/mr_center2_k05']:(RUN/s).mkdir(parents=True,exist_ok=True)
 config={'experiment':'E17','hypothesis':'GT-box-only training causes a source segmentation accuracy gap on actual detector boxes; adding actual MR detector-positive crops can reduce it','precondition':'source helper meanDice on original detector positives must be at least.02 below original GT-box source meanDice before any new S training','source':'existing E04 GT-jitter bank plus MR source score.3/top5 original detector positives from E12 cache; no validation-error backfill','development_split':'same E05 case groups used by E10 helper, which excludes these cases; source upstream detector is in-sample and this limitation persists','proposed_training':'only if source gap passes: same architecture/optimizer/batch32 and fixed13dev+13finalepochs; no detector threshold change','adoption':'MR40 official all-six Pareto gain versusE14 and retain53matched lesions; otherwise keepE04 MR segmenter','CT_deployment_unchanged':True,'budget_hours':3,'no_MR40_or_CT5_fit':True};write_json(RUN/'config.json',config)
 old=[json.loads(s) for s in (PARENT/'features/train_records.jsonl').read_text().splitlines()];fr=[json.loads(s) for s in (F/'features/records.jsonl').read_text().splitlines()];images=np.load(F/'features/images.npy',mmap_mode='r');groups={};pools={}
 for i,r in enumerate(fr):
  cid=r['case_id']
  if r['augmentation'] or not r['y']:continue
  assert '_mr_' in cid and 'center2' not in cid
  if cid not in pools:
   folder=P/('artifacts/fold1_source_center5_epoch60_20260909' if 'center5' in cid else 'artifacts/fold1_eval_center1_epoch60');bx,sc,_=load_boxes(folder/f'{cid}_boxes.pkl');pools[cid]={v[0] for v in select_candidates(bx,sc)}
  if r['original_index'] in pools[cid]:groups.setdefault(cid,[]).append((i,r))
 caches=[]
 for n,(cid,items) in enumerate(sorted(groups.items()),1):
  dest=RUN/f'features/cases/{cid}.npz';meta=dest.with_suffix('.json')
  if dest.exists() and meta.exists():caches.append((dest,meta));continue
  gt,ga,shape=load_nifti(DATA/f'location_masks/{cid}.nii.gz');aff,ish=load_nifti_geometry(DATA/f'images/{cid}_0000.nii.gz');identity,bound=nearest_label_identity(aff,ga,shape);assert shape==ish and identity;comps=component_records_fast(gt);xx=[];yy=[];rr=[]
  for i,r in items:
   low,high=np.array(r['low']),np.array(r['high']);choices=[]
   for comp in comps:
    count=np.all((comp['coords']>=low)&(comp['coords']<high),axis=1).sum()
    if not count:continue
    lo=comp['coords'].min(0)-.5;hi=comp['coords'].max(0)+.5;inter=np.maximum(0,np.minimum(high,hi)-np.maximum(low,lo)).prod();iou=inter/max(1e-12,(high-low).prod()+(hi-lo).prod()-inter);choices.append((iou,count,comp))
   assert choices;comp=max(choices,key=lambda a:(a[0],a[1]))[2];lo=comp['coords'].min(0);hi=comp['coords'].max(0)+1;small=np.zeros(tuple(hi-lo),np.uint8);small[tuple((comp['coords']-lo).T)]=1;small=np.pad(small,1);lo-=1;coords=crop_coordinates(low,high);target=map_coordinates(small,(coords-lo[:,None,None,None]).reshape(3,-1),order=0,mode='constant',cval=0,prefilter=False).reshape(32,32,32);valid=np.all((coords>=0)&(coords<=(np.array(shape)-1)[:,None,None,None]),axis=0);target=(target*SUPPORT*valid).astype(np.uint8)
   if not target.any():continue
   xx.append(images[i,0]);yy.append(target);rr.append({'case_id':cid,'source_class_id':comp['class_id'],'component_id':comp['component_id'],'sample_index':1000+r['original_index'],'low':r['low'],'high':r['high'],'detector_index':r['original_index'],'score':r['score'],'source_filter_record_index':i,'source':'actual_detector_positive','nearest_label_identity':identity,'max_voxel_error_per_axis':bound})
  np.savez_compressed(dest,x=np.asarray(xx,np.float16),y=np.asarray(yy,np.uint8));write_json(meta,rr);caches.append((dest,meta));print('MR_detector_crop',n,len(groups),cid,len(rr),flush=True)
 count=sum(len(json.loads(meta.read_text())) for _,meta in caches);oldx=np.load(PARENT/'features/images.npy',mmap_mode='r');oldy=np.load(PARENT/'features/targets.npy',mmap_mode='r');x=np.lib.format.open_memmap(RUN/'features/images.npy',mode='w+',dtype=np.float16,shape=(2493+count,32,32,32));y=np.lib.format.open_memmap(RUN/'features/targets.npy',mode='w+',dtype=np.uint8,shape=x.shape);x[:2493]=oldx;y[:2493]=oldy;mapping=np.load(PARENT/'features/rows.npz');idx=mapping['crop_index'].tolist();mir=mapping['mirrored'].tolist();records=old.copy();offset=2493
 for dest,meta in caches:
  a=np.load(dest);rr=json.loads(meta.read_text());x[offset:offset+len(rr)]=a['x'];y[offset:offset+len(rr)]=a['y']
  for j,r in enumerate(rr):
   for flip in [False,True]:idx.append(offset+j);mir.append(flip);records.append({**r,'view':'mirror' if flip else 'original'})
  offset+=len(rr)
 x.flush();y.flush();assert offset==len(x) and np.isfinite(x).all();np.savez_compressed(RUN/'features/rows.npz',crop_index=idx,mirrored=mir);(RUN/'features/train_records.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n');split=json.loads((P/'artifacts/astra6_e05_learned_location_splits_20260909/source_split.json').read_text());dev=set(split['development_cases']);tr=[i for i,r in enumerate(records) if r['case_id'] not in dev];dv=[i for i,r in enumerate(records) if i>=4986 and r['case_id'] in dev and r['view']=='original'];gtdev=[i for i,r in enumerate(records[:4986]) if '_mr_' in r['case_id'] and r['case_id'] in dev and r['view']=='original' and r['sample_index']==0];assert dv and gtdev and not {records[i]['case_id'] for i in tr}&dev
 write_json(RUN/'source_split.json',{'development_cases':sorted(dev),'train_rows':tr,'development_detector_rows':dv,'development_GT_rows':gtdev});write_json(RUN/'features/SOURCE_READY.json',{'new_detector_crops':count,'n_crops':len(x),'n_rows':len(records),'source_detector_cases':len(groups),'images_sha256':sha256_file(RUN/'features/images.npy'),'targets_sha256':sha256_file(RUN/'features/targets.npy'),'records_sha256':sha256_file(RUN/'features/train_records.jsonl')});baseline={'GT_boxes':evaluate(HELPER,gtdev),'detector_boxes':evaluate(HELPER,dv)};gap=baseline['GT_boxes']['crop_Dice_mean']-baseline['detector_boxes']['crop_Dice_mean'];write_json(RUN/'evaluation/SOURCE_PRECONDITION.json',{'baseline':baseline,'Dice_gap':gap,'proceed_to_training':gap>=.02,'no_MR40_access':True});print('SOURCE_PRECONDITION',gap,gap>=.02,flush=True)
if __name__=='__main__':main()
