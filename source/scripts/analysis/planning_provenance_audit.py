"""Read-only provenance audit: legacy supervised planning versus research splits."""
import json,pickle,hashlib,datetime
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[2];V=P.parent;R=P/'artifacts/research_audit_20260909'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return pickle.loads(p.read_bytes())
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def main():
 mr=json.loads((P/'artifacts/astra6_e23_MR_oof_candidates_20260909/source_split.json').read_text())
 ct=json.loads((P/'artifacts/astra6_e16_CT_expanded_segmentation_20260909/source_split.json').read_text())
 out={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Known internal research holdout planning exposure; no hidden competition test access established.','modalities':{}}
 for mod,task,fold,held,expected in [('MR','Task030FG_TopAneuMR',1,mr['comparison_excluded'],248),('CT','Task031FG_TopAneuCT',2,ct['fixed_CT5'],136)]:
  path=V/f'nndet_data/{task}/preprocessed/D3V001_3d.pkl';deployed=V/f'nndet_models/{task}/RetinaUNetV001_D3V001_3d/fold{fold}/plan.pkl';plan=load(path);dp=load(deployed);props=plan['dataset_properties']['instance_props_per_patient'];owners=[];boxes=[];tf=plan['transpose_forward'];target=np.asarray(plan['target_spacing_transposed'])
  for c,q in props.items():
   b=q['boxes']
   if isinstance(b,list) or not b.size:continue
   low=b[:,[0,1,4]][:,tf];high=b[:,[2,3,5]][:,tf];scale=np.asarray(q['original_spacing'])[tf]/target
   low=low*scale;high=high*scale;scaled=np.stack([low[:,0],low[:,1],high[:,0],high[:,1],low[:,2],high[:,2]],axis=1)
   boxes.append(scaled);owners.extend([c]*len(b))
  boxes=np.concatenate(boxes).astype(np.float32);dims=boxes[:,[2,3,5]]-boxes[:,[0,1,4]];lo=np.percentile(dims,.5,axis=0);hi=np.percentile(dims,99.5,axis=0);keep=np.all((dims>lo)&(dims<hi),axis=1);assert int(keep.sum())==expected,(mod,int(keep.sum()))
  logs=V/f'nndet_data/{task}/logging.log';lines=[{'line':i+1,'text':s} for i,s in enumerate(logs.read_text().splitlines()) if 'Determined Anchors:' in s or 'boxes remaining for anchor' in s]
  assert dp['anchors']==plan['anchors'];assert dp['normalization_schemes']==plan['normalization_schemes'];assert set(dp['dataset_properties']['instance_props_per_patient'])==set(props)
  intensity=plan['dataset_properties']['intensity_properties'];assert str(dp['dataset_properties']['intensity_properties'])==str(intensity)
  cases={c:{'in_legacy_planning':c in props,'GT_boxes_entering_anchor_quantiles':sum(o==c for o in owners),'GT_boxes_retained_for_anchor_optimization':sum(o==c and bool(k) for o,k in zip(owners,keep))} for c in held};assert all(q['in_legacy_planning'] for q in cases.values())
  entry={'data_plan':str(path),'data_plan_sha256':sha(path),'deployed_plan':str(deployed),'deployed_plan_sha256':sha(deployed),'deployed_anchor_normalization_case_membership_match':True,'planning_case_count':len(props),'planning_cases':sorted(props),'anchor_input_boxes':len(boxes),'anchor_retained_boxes':int(keep.sum()),'research_comparison_cases':cases,'anchors':plan['anchors'],'normalization':plan['normalization_schemes'],'use_mask_for_norm':plan['use_mask_for_norm'],'global_GT_intensity_statistics_used_by_normalization':mod=='CT','intensity_properties':intensity,'historical_planning_log':str(logs),'historical_planning_log_sha256':sha(logs),'historical_anchor_log_evidence':lines}
  if mod=='MR':
   e=P/'artifacts/astra6_e23_MR_oof_candidates_20260909/data/Task130FG_TopAneuMR_OOF/preprocessed/D3V001_3d.pkl';assert sha(e)==sha(path);entry['E23_plan_identical_sha256']=sha(e);entry['E23_fold_own_heldout_cases_in_legacy_plan']={str(i):len(set(f['val'])&set(props)) for i,f in enumerate(mr['folds'])}
  out['modalities'][mod]=entry
 base=V/'external/nnDetection/nndet';refs={'planning/architecture/boxes/c002.py':'process_properties consumes all case GT boxes; _plan_anchors optimizes IoU using these boxes.','planning/architecture/boxes/base.py':'filter_boxes computes global GT size quantiles before anchor optimization.','planning/properties/intensity.py':'collects GT foreground intensity across analyzer.case_ids.','preprocessing/preprocessor.py':'normalize_ct consumes global GT mean/std/percentiles; nonCT uses per-image normalization.'}
 out['source_evidence']={str(base/k):{'sha256':sha(base/k),'finding':v} for k,v in refs.items()};out['conclusion']='Per-case gradient-fit exclusion holds separately, but legacy supervised planning includes research comparison and OOF cases. Shared-plan paired comparisons are developmental evidence only. Removing cached metadata cannot undo learned planning exposure.'
 (R/'PLANNING_PROVENANCE_AUDIT.json').write_text(json.dumps(out,indent=2,default=serial)+'\n')
 print(json.dumps({m:{'cases':v['planning_case_count'],'boxes':v['anchor_input_boxes'],'retained':v['anchor_retained_boxes'],'comparison_GT_boxes':sum(c['GT_boxes_entering_anchor_quantiles'] for c in v['research_comparison_cases'].values()),'comparison_retained':sum(c['GT_boxes_retained_for_anchor_optimization'] for c in v['research_comparison_cases'].values())} for m,v in out['modalities'].items()},indent=2),flush=True)
if __name__=='__main__':main()
