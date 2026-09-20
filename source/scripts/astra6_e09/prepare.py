"""CT-only classifier adaptation: add42 eligible center2 CT source cases, never CT5 or MR40."""
from pathlib import Path
import json,time
import numpy as np
from scripts.astra6_e01.e01_common import P,DATA,load_nifti,load_nifti_geometry,compute_feature,sha256_file,write_json
from scripts.astra6_e01.dataset import jitter_boxes
from scripts.astra6_e03.run_e03 import component_records_fast
from scripts.astra6_e02.run_e02 import multiscale,extended_schema,E01
from scripts.delivery.geometry import vessel_geometry_fast
BASE=P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z';RUN=P/'artifacts/astra6_e09_CT_expanded_location_20260909'

def main():
 for s in ['features/cases','model','evaluation','logs','predictions_ct']:(RUN/s).mkdir(parents=True,exist_ok=True)
 test=json.loads((P/'artifacts/ct_independent5_20260909/cohort.json').read_text())['cases'];allct=sorted(f.name.removesuffix('_0000.nii.gz') for f in (DATA/'images').glob('*center2_ct*_0000.nii.gz'));ids=[c for c in allct if c not in test];assert len(ids)==42 and not set(ids)&set(test)
 shuffled=ids.copy();np.random.default_rng(20260909).shuffle(shuffled);dev=set(shuffled[:8]);schema=json.loads((E01/'feature_schema.json').read_text());locpair={int(k):int(v) for k,v in schema['location_lr_pair'].items()};_,vpair,perm=extended_schema(schema)
 write_json(RUN/'source_split.json',{'additional_eligible_CT_cases':ids,'source_development_cases':sorted(dev),'fixed_CT5':test,'existing_source':str(BASE/'features/train_records.jsonl'),'inference_modality':'CT only','MR_weights_unchanged':True,'CT_comparison_is_not_LOCO':True,'case_ID_disjoint':True,'patient_linkage_limitation':'no additional patient-level linkage beyond official IDs'})
 write_json(RUN/'config.json',{'experiment':'E09','hypothesis':'additional eligible official CT source labels improve location classification on CT','change':'only add CT source cases; same943 features, ExtraTrees512/depth16/leaf2/max_features.5/seed20260905, same augmentation; MR uses old model','source_vessels':'official provided TopBrain-organizer predicted silver masks, same source convention as E02','inference_vessels':'predictedTA36 only; never provided vessel masks at inference','source_selection':'42 center2CT outside fixed5; all old218 source cases reused','budget_hours':2,'checkpoint_every_trees':64,'adoption':'CT5 all-six Pareto improvement, with tiny-sample limitation','no_CT5_or_MR40_fit':True})
 for n,cid in enumerate(ids,1):
  cache=RUN/f'features/cases/{cid}.npz'
  if cache.exists() and cache.with_suffix('.json').exists():continue
  aff,shape=load_nifti_geometry(DATA/f'images/{cid}_0000.nii.gz');gt,ga,gs=load_nifti(DATA/f'location_masks/{cid}.nii.gz');assert shape==gs and np.allclose(aff,ga,atol=1e-4)
  comps=component_records_fast(gt);records=[];features=[];rng=np.random.default_rng(20260905+sum(map(ord,cid)))
  geom=vessel_geometry_fast(DATA/f'vessel_masks/{cid}.nii.gz',shape,aff) if comps else None
  for comp in comps:
   comp['low']=comp['coords'].min(0).astype(float)-.5;comp['high']=comp['coords'].max(0).astype(float)+.5;jitter=jitter_boxes(comp,aff,shape,rng)
   for i,(lo,hi) in enumerate(jitter):
    extra=multiscale(geom,aff,lo,hi)
    for mirror in [False,True]:
     y=locpair[comp['class_id']] if mirror else comp['class_id'];features.append(np.concatenate([compute_feature(geom,aff,lo,hi,'CT',mirror,vpair),extra[perm] if mirror else extra]));records.append({'case_id':cid,'source_class_id':comp['class_id'],'class_id':y,'component_id':comp['component_id'],'sample_index':i,'view':'mirror' if mirror else 'original','low':lo.tolist(),'high':hi.tolist(),'source_development':cid in dev})
  np.savez_compressed(cache,X=np.asarray(features,np.float32).reshape(-1,943));cache.with_suffix('.json').write_text('\n'.join(json.dumps(r) for r in records)+'\n');print(cid,n,42,'components',len(comps),flush=True)
 old=np.load(BASE/'features/train.npz');oldrows=[json.loads(s) for s in (BASE/'features/train_records.jsonl').read_text().splitlines()];newrows=[];xx=[old['X']]
 for cid in ids:
  xx.append(np.load(RUN/f'features/cases/{cid}.npz')['X']);newrows.extend(json.loads(s) for s in (RUN/f'features/cases/{cid}.json').read_text().splitlines() if s)
 records=oldrows+newrows;X=np.concatenate(xx);y=np.array([r['class_id'] for r in records],np.int16);assert len(records)==len(X) and np.isfinite(X).all() and not any(r['case_id'] in test or 'center2_mr' in r['case_id'] for r in records)
 np.savez_compressed(RUN/'features/train.npz',X=X,y=y);(RUN/'features/train_records.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n');write_json(RUN/'features/READY.json',{'rows':len(records),'old_rows':len(oldrows),'new_rows':len(newrows),'positive_source_cases':len({r['case_id'] for r in records}),'features_sha256':sha256_file(RUN/'features/train.npz'),'records_sha256':sha256_file(RUN/'features/train_records.jsonl'),'additional_label_sha256':{c:sha256_file(DATA/f'location_masks/{c}.nii.gz') for c in ids},'additional_vessel_sha256':{c:sha256_file(DATA/f'vessel_masks/{c}.nii.gz') for c in ids}});print('SOURCE_READY',flush=True)
if __name__=='__main__':main()
