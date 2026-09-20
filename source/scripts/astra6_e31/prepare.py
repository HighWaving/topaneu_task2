"""Persistent case-wise matched crop banks; GT read exclusively for S training targets."""
import json,os,time
import numpy as np,nibabel as nib
from scripts.astra6_e31.common import *
from scripts.astra6_e01.e01_common import DATA,write_json,sha256_file
from scripts.astra6_e04.run_e04 import load_image,component_records_fast
from scripts.astra6_e29.prepare import group
E16=P/'artifacts/astra6_e16_CT_expanded_segmentation_20260909';F=P/'artifacts/astra6_e25_oof_shape_fp_filter_20260909';SRC=P/'artifacts/astra6_e29_staged_anatomy_location_20260910'
def main():
 if (RUN/'features/READY.json').exists():return
 (RUN/'features/cases').mkdir(parents=True,exist_ok=True)
 split=json.loads((SRC/'source_split.json').read_text());dev=set(split['development_cases']);devgroups={group(c) for c in dev}
 source=[json.loads(s) for s in (SRC/'features/records.jsonl').read_text().splitlines()];val={(source[i]['case_id'],source[i]['candidate_index']):source[i]['component_index'] for i in split['validation_rows']}
 gtrows=[json.loads(s) for s in (E16/'features/train_records.jsonl').read_text().splitlines()];gtrows=[r for r in gtrows if r['view']=='original']
 oof=[json.loads(s) for s in (F/'features/records.jsonl').read_text().splitlines()];oof=[r for r in oof if r['y'] and len(r['matched_components'])==1]
 cases=sorted({r['case_id'] for r in gtrows+oof});assert all('center2_mr' not in c for c in cases)
 dsplit=json.loads((P/'artifacts/astra6_e23_MR_oof_candidates_20260909/source_split.json').read_text())['folds'][1]
 assert not devgroups & {group(c) for c in dsplit['train']}
 assert all(source[i]['OOF_fold']==1 for i in split['validation_rows'])
 rows=[];normalization={}
 for n,cid in enumerate(cases,1):
  started=time.monotonic()
  meta=RUN/f'features/cases/{cid}.json'
  if meta.exists():
   saved=json.loads(meta.read_text());rows.extend(saved['rows']);normalization[cid]=saved['normalization'];continue
  image,aff,norm=load_image(cid);gtimg=nib.load(str(DATA/f'location_masks/{cid}.nii.gz'));assert gtimg.shape==image.shape and np.allclose(gtimg.affine,aff,atol=1e-4);gt=np.asanyarray(gtimg.dataobj);components=component_records_fast(gt)
  vesselimg=nib.load(str(DATA/f'vessel_masks/{cid}.nii.gz'));assert vesselimg.shape==image.shape and np.allclose(vesselimg.affine,aff,atol=1e-4);vessel=np.asanyarray(vesselimg.dataobj);assert vessel.min()>=0 and vessel.max()<=36
  cr=[]
  for r in [r for r in gtrows if r['case_id']==cid]:
   base=next(b for b in gtrows if b['case_id']==cid and b['component_id']==r['component_id'] and b['source_class_id']==r['source_class_id'] and b['sample_index']==0)
   matches=[j for j,c in enumerate(components) if c['class_id']==r['source_class_id'] and np.array_equal(c['coords'].min(0)-.5,base['low']) and np.array_equal(c['coords'].max(0)+.5,base['high'])];assert len(matches)==1
   cr.append({'case_id':cid,'component_index':matches[0],'kind':'GT','low':r['low'],'high':r['high'],'use_train':group(cid) not in devgroups,'use_val':False,'sample_index':r['sample_index']})
  for r in [r for r in oof if r['case_id']==cid]:
   ci=r['matched_components'][0];key=(cid,r['original_index']);is_val=key in val
   if is_val:assert val[key]==ci
   cr.append({'case_id':cid,'component_index':ci,'kind':'OOF','low':r['low'],'high':r['high'],'candidate_index':r['original_index'],'OOF_fold':r['fold'],'operating':r['deployed'],'use_train':r['fold']==1 and group(cid) not in devgroups,'use_val':is_val})
  arrays={a:{'x':[],'y':[],'origin':[],'step':[]} for a in ['normalized','physical']}
  for j,r in enumerate(cr):
   comp=components[r['component_index']]['coords'];r.update(case_row=j,class_id=int(components[r['component_index']]['class_id']),GT_voxels=len(comp),image_shape=list(image.shape),GT_volume_mm3=float(len(comp)*abs(np.linalg.det(aff[:3,:3]))))
   for arm,z in arrays.items():
    x,origin,step=input_crop(image,vessel,r['low'],r['high'],aff,arm);y=target_crop(comp,origin,step);assert np.isfinite(x).all();z['x'].append(x);z['y'].append(y);z['origin'].append(origin);z['step'].append(step)
    r[arm+'_target_voxels']=int(y.sum())
  for arm,z in arrays.items():
   path=RUN/f'features/cases/{cid}_{arm}.npz';tmp=path.with_suffix('.tmp')
   with tmp.open('wb') as handle:np.savez_compressed(handle,**{k:np.asarray(v) for k,v in z.items()})
   os.replace(tmp,path)
  write_json(meta,{'rows':cr,'normalization':norm,'vessel_source':str(DATA/f'vessel_masks/{cid}.nii.gz'),'vessel_mask_usage':'organizer predicted vessel, not GT vessel','image_affine':aff.tolist()});rows.extend(cr);normalization[cid]=norm
  print('E31 crops',n,len(cases),cid,len(cr),'seconds',round(time.monotonic()-started,2),flush=True)
  del image,gt,vessel,components,arrays
 assert sum(r['use_val'] for r in rows)==26
 tr=[i for i,r in enumerate(rows) if r['use_train']];dv=[i for i,r in enumerate(rows) if r['use_val']]
 assert not {group(rows[i]['case_id']) for i in tr}&devgroups
 for arm in ['normalized','physical']:
  dest=RUN/arm/'features';dest.mkdir(parents=True,exist_ok=True)
  x=np.lib.format.open_memmap(dest/'images.npy',mode='w+',dtype=np.float16,shape=(len(rows),3,N,N,N));y=np.lib.format.open_memmap(dest/'targets.npy',mode='w+',dtype=np.uint8,shape=(len(rows),N,N,N));origin=np.zeros((len(rows),3));step=origin.copy();offset=0
  for cid in cases:
   z=np.load(RUN/f'features/cases/{cid}_{arm}.npz');n=len(z['x']);x[offset:offset+n]=z['x'];y[offset:offset+n]=z['y'];origin[offset:offset+n]=z['origin'];step[offset:offset+n]=z['step'];offset+=n
  assert offset==len(rows);x.flush();y.flush();np.savez(dest/'geometry.npz',origin=origin,step=step)
  write_json(dest/'READY.json',{'rows':len(rows),'images_sha256':sha256_file(dest/'images.npy'),'targets_sha256':sha256_file(dest/'targets.npy'),'geometry_sha256':sha256_file(dest/'geometry.npz')});del x,y
 (RUN/'features/records.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
 write_json(RUN/'source_split.json',{'train_rows':tr,'validation_rows':dv,'final_rows':list(range(len(rows))),'development_cases':sorted(dev),'D1_gradient_training_cases':dsplit['train'],'GT_source_cases':sorted({r['case_id'] for r in gtrows}),'source_OOF_only_fold1':True,'source_fit_cases':sorted({rows[i]['case_id'] for i in tr})})
 write_json(RUN/'features/READY.json',{'n_cases':len(cases),'n_rows':len(rows),'train_rows':len(tr),'validation_rows':len(dv),'records_sha256':sha256_file(RUN/'features/records.jsonl'),'config_sha256':sha256_file(RUN/'config.json'),'known_legacy_planning_and_vessel_history_limits':True})
 print('E31 ALL CROPS READY',len(rows),len(tr),len(dv),flush=True)
if __name__=='__main__':main()
