"""Source-only joint-image anatomy crops, guarded by completed E23/E25."""
import json,hashlib
from pathlib import Path
import numpy as np,torch
from scripts.astra6_e28.common import P,RUN,OOF,BASE,SEED
from scripts.astra6_e28.representation import resample_ras,shape_to_ras,anatomy_center_indices
from scripts.astra6_e04.run_e04 import Segmenter,load_image,crop_volume,PRIOR,component_records_fast
from scripts.astra6_e01.e01_common import DATA,load_nifti,load_boxes,select_candidates,box_to_native_bounds,sha256_file,write_json,compute_feature
from scripts.astra6_e12.common import nearest_label_identity
from scripts.astra6_e25.prepare import group
from scripts.astra6_e25.source_candidate_bias import wide_indices
from scripts.astra6_e02.run_e02 import multiscale,extended_schema,E01
from scripts.delivery.geometry import vessel_geometry_fast

def main():
 torch.set_num_threads(2)
 assert (P/'artifacts/astra6_e25_oof_shape_fp_filter_20260909/evaluation/DECISION.json').exists(),'Keep E23/E25 ahead of this formal branch'
 if (RUN/'features/READY.json').exists():return
 config=json.loads((RUN/'config.json').read_text());part=config['source_partition'];fit=set(part['C_fit_cases']);dev=set(part['epoch_selection_cases']);allsource=set(part['final_eligible_cases']);assert not {group(c) for c in fit}&{group(c) for c in dev}
 folds=json.loads((OOF/'source_split.json').read_text())['folds'];locks=[json.loads((OOF/f'checkpoints/fold{f}_OOF_COMPLETE.json').read_text()) for f in [0,1]];owner={c:f for f,l in enumerate(locks) for c in l['cases']};models={};modelhashes={}
 for f in [0,1]:
  assert locks[f]['cases']==folds[f]['val'] and locks[f]['source_split_sha256']==sha256_file(OOF/'source_split.json')
  sp=OOF/f'segmenters/fold{f}/source_split.json';ss=json.loads(sp.read_text());assert not {group(c) for c in ss['training_cases']}&{group(c) for c in folds[f]['val']};weight=sp.parent/'model/final_last.pt';lock=json.loads(weight.with_name('LOCKED.json').read_text());assert sha256_file(weight)==lock['sha256'] and sha256_file(sp)==lock['source_split_sha256'];model=Segmenter().cuda().eval();model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=False)['state_dict']);models[f]=model;modelhashes[str(f)]=lock['sha256']
 (RUN/'features/cases').mkdir(parents=True,exist_ok=True);schema=json.loads((E01/'feature_schema.json').read_text());_,vp,_=extended_schema(schema);records=[]
 for ci,cid in enumerate(sorted(allsource),1):
  file=RUN/f'features/cases/{cid}.npz';meta=file.with_suffix('.json');f=owner[cid]
  if file.exists() and meta.exists():
   cached=json.loads(meta.read_text());assert cached['helper_sha256']==modelhashes[str(f)] and cached['config_sha256']==sha256_file(RUN/'config.json') and cached['prepare_code_sha256']==sha256_file(Path(__file__));records.extend(cached['rows']);continue
  arr,aff,_=load_image(cid);gt,ga,_=load_nifti(DATA/f'location_masks/{cid}.nii.gz');vessel,va,_=load_nifti(DATA/f'vessel_masks/{cid}.nii.gz');assert gt.shape==vessel.shape==arr.shape and nearest_label_identity(aff,ga,arr.shape)[0] and np.allclose(aff,va,atol=1e-4);assert vessel.min()>=0 and vessel.max()<=36 and gt.min()>=0 and gt.max()<=52;comps=component_records_fast(gt);rng=np.random.default_rng(SEED+int(hashlib.sha256(cid.encode()).hexdigest()[:8],16));items=[]
  # Training-only GT-centered/jittered crops broaden class supervision. Neither
  # GT mask nor GT vessel is an inference input. Dev decisions use OOF boxes only.
  for gi,c in enumerate(comps):
   lo=c['coords'].min(0)-.5;hi=c['coords'].max(0)+.5
   for jitter in [0,1,2]:
    delta=np.linalg.solve(aff[:3,:3],rng.uniform(-3,3,3)) if jitter else np.zeros(3);items.append({'class':int(c['class_id']),'component_index':gi,'low':(lo+delta).tolist(),'high':(hi+delta).tolist(),'kind':'GT_training_crop','operating':False,'use_C_fit':cid in fit,'use_C_validation':False})
  path=OOF/f'oof_boxes/fold{f}/{cid}_boxes.pkl';assert sha256_file(path)==locks[f]['boxes_sha256'][cid];bx,sc,_=load_boxes(path);operating={i for i,s,l,h in select_candidates(bx,sc)}
  for idx in wide_indices(sc):
   score=float(sc[idx])
   lo,hi=box_to_native_bounds(bx[idx]);covered=[j for j,c in enumerate(comps) if np.all((c['coords']>=lo)&(c['coords']<hi),axis=1).mean()>=.1]
   if len(covered)!=1:continue
   gi=covered[0];items.append({'class':int(comps[gi]['class_id']),'component_index':gi,'low':lo.tolist(),'high':hi.tolist(),'kind':'OOF_positive','candidate_index':int(idx),'score':float(score),'operating':idx in operating,'use_C_fit':cid in fit,'use_C_validation':cid in dev and idx in operating})
  # Dense anatomy is image-only supervision; no aneurysm class for these crops.
  coords=np.argwhere(vessel>0)
  if len(coords):
   center_labels=vessel[tuple(coords.T)]
   for j,center_index in enumerate(anatomy_center_indices(center_labels,rng)):
    center=coords[center_index].astype(float);extent=6/np.linalg.norm(aff[:3,:3],axis=0);items.append({'class':-1,'component_index':-1,'low':(center-extent/2).tolist(),'high':(center+extent/2).tolist(),'kind':'silver_anatomy_only','anatomy_center_sampling':'voxel_proportional' if j==0 else 'uniform_present_label','anatomy_center_label':int(center_labels[center_index]),'operating':False,'use_C_fit':False,'use_C_validation':False})
  local=[];context=[];shapes=[];silver=[];rr=[];geometry=vessel_geometry_fast(DATA/f'vessel_masks/{cid}.nii.gz',arr.shape,aff) if cid in dev else None
  for j,item in enumerate(items):
   lo,hi=np.array(item['low']),np.array(item['high']);center=(lo+hi)/2;local.append(resample_ras(arr,aff,center,24,48).astype(np.float16));context.append(resample_ras(arr,aff,center,80,64).astype(np.float16));silver.append(resample_ras(vessel,aff,center,24,48,0).astype(np.uint8))
   if item['class']>0:
    x=np.stack([crop_volume(arr,lo,hi,1),PRIOR])[None].astype(np.float32)
    with torch.inference_mode():prob=models[f](torch.from_numpy(x).cuda()).sigmoid().cpu().numpy()[0,0]
    shapes.append(shape_to_ras(prob,lo,hi,aff).astype(np.float16))
   else:shapes.append(np.zeros((48,48,48),np.float16))
   row={**item,'case_id':cid,'case_row':j,'OOF_fold':f,'use_anatomy_fit':cid not in dev,'use_final':True}
   if item['kind']=='silver_anatomy_only':row['center_label_present_after_resampling']=bool(np.any(silver[-1]==item['anatomy_center_label']))
   if item['use_C_validation']:row['baseline_geometry_features']=np.concatenate([compute_feature(geometry,aff,lo,hi,'MR',False,vp),multiscale(geometry,aff,lo,hi)]).astype(np.float32).tolist()
   rr.append(row)
  np.savez_compressed(file,local=np.asarray(local,np.float16).reshape(-1,48,48,48),context=np.asarray(context,np.float16).reshape(-1,64,64,64),shape=np.asarray(shapes,np.float16).reshape(-1,48,48,48),silver=np.asarray(silver,np.uint8).reshape(-1,48,48,48));write_json(meta,{'config_sha256':sha256_file(RUN/'config.json'),'prepare_code_sha256':sha256_file(Path(__file__)),'rows':rr,'helper_sha256':modelhashes[str(f)],'D_checkpoint_sha256':locks[f]['checkpoint_sha256'],'image_sha256':sha256_file(DATA/f'images/{cid}_0000.nii.gz'),'GT_location_target_sha256':sha256_file(DATA/f'location_masks/{cid}.nii.gz'),'vessel_silver_sha256':sha256_file(DATA/f'vessel_masks/{cid}.nii.gz'),'vessel_is_organizer_prediction':True,'GT_lesions':len(comps)});records.extend(rr);print('E28 source',ci,len(allsource),cid,len(rr),flush=True)
  del arr,gt,vessel,coords,comps,local,context,shapes,silver,geometry
 (RUN/'features/records.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n')
 train=[i for i,r in enumerate(records) if r['use_C_fit'] or (r['class']<0 and r['use_anatomy_fit'])];eligible=[i for i,r in enumerate(records) if r['use_C_validation']];per_component={}
 for i in eligible:
  r=records[i];key=(r['case_id'],r['component_index']);previous=per_component.get(key)
  if previous is None or (r['score'],-r['candidate_index'])>(records[previous]['score'],-records[previous]['candidate_index']):per_component[key]=i
 validation=sorted(per_component.values());assert train and validation and not {group(records[i]['case_id']) for i in train}&{group(records[i]['case_id']) for i in validation}
 arrays={key:np.lib.format.open_memmap(RUN/f'features/{key}.npy',mode='w+',dtype=dtype,shape=(len(records),size,size,size)) for key,dtype,size in [('local',np.float16,48),('context',np.float16,64),('shape',np.float16,48),('silver',np.uint8,48)]};offset=0
 for cid in sorted(allsource):
  with np.load(RUN/f'features/cases/{cid}.npz') as z:
   n=len(z['local'])
   for key,a in arrays.items():a[offset:offset+n]=z[key]
   offset+=n
 assert offset==len(records)
 for a in arrays.values():a.flush()
 write_json(RUN/'source_split.json',{'train_rows':train,'validation_rows':validation,'final_rows':list(range(len(records))),'training_C_cases':sorted(fit),'development_cases':sorted(dev),'validation_pairing':'One unambiguous operating OOF box per GT component, highest fixed D score; no duplicate-component inflation','source_only':True});write_json(RUN/'features/READY.json',{'rows':len(records),'source_cases':len(allsource),'validation_candidates':len(validation),'validation_GT_components':len({(records[i]['case_id'],records[i]['component_index']) for i in validation}),'records_sha256':sha256_file(RUN/'features/records.jsonl'),'source_split_sha256':sha256_file(RUN/'source_split.json'),'config_sha256':sha256_file(RUN/'config.json'),'helper_sha256':modelhashes,'arrays_sha256':{k:sha256_file(RUN/f'features/{k}.npy') for k in arrays},'no_MR40_CT5_fit':True})
if __name__=='__main__':
 main()
 from scripts.analysis.research_lineage_audit import main as audit_lineage
 audit_lineage()
