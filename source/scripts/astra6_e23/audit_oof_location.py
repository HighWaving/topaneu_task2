"""Source-only held-out C audit on genuinely OOF detector proposals.
No fitting, holdout inference, candidate correction, or vessel GT input.
"""
from pathlib import Path
import json,time
import numpy as np,joblib
from scripts.astra6_e23.prepare import P,RUN
from scripts.astra6_e01.e01_common import DATA,load_boxes,select_candidates,load_nifti,load_nifti_geometry,compute_feature,write_json,sha256_file
from scripts.astra6_e02.run_e02 import multiscale,extended_schema,E01
from scripts.astra6_e03.run_e03 import component_records_fast
from scripts.delivery.geometry import vessel_geometry_fast
OUT=RUN/'location_source_audit';BASE=P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z';OLD=P/'artifacts/astra6_e05_learned_location_splits_20260909'
def main():
 OUT.mkdir(exist_ok=True);locks=[json.loads((RUN/f'checkpoints/fold{i}_OOF_COMPLETE.json').read_text()) for i in [0,1]];owner={c:i for i,a in enumerate(locks) for c in a['cases']};dev=set(json.loads((OLD/'source_split.json').read_text())['development_cases']);ids=sorted(c for c in dev if c in owner);assert ids and not any('center2' in c for c in ids);model=joblib.load(OLD/'model/development_baseline.joblib');old=np.load(BASE/'features/train.npz');records=[json.loads(s) for s in (BASE/'features/train_records.jsonl').read_text().splitlines()];schema=json.loads((E01/'feature_schema.json').read_text());_,vpair,_=extended_schema(schema)
 config={'experiment':'E23 source location audit','hypothesis':'Properly OOF detector boxes may expose source location degradation hidden by earlier in-sample detector box audit.','training_trigger':'At least20 paired lesions and OOF-box C has at least2 fewer correct than GT-box C, using the same source-heldout C model.','no_new_model_fit':True,'no_MR40_CT5_read':True,'anatomy':'Organizer-provided predicted silver vessel masks; same943features as E02, notGTvessel.','source_baseline_sha256':sha256_file(OLD/'model/development_baseline.joblib')};write_json(OUT/'config.json',config);rows=[];case_summary={}
 for n,cid in enumerate(ids,1):
  marker=OUT/f'{cid}.json'
  if marker.exists():
   a=json.loads(marker.read_text());rows.extend(a['rows']);case_summary[cid]=a['summary'];continue
  f=RUN/f'oof_boxes/fold{owner[cid]}/{cid}_boxes.pkl';assert sha256_file(f)==locks[owner[cid]]['boxes_sha256'][cid];bx,sc,_=load_boxes(f);selected=select_candidates(bx,sc);gt,_,_=load_nifti(DATA/f'location_masks/{cid}.nii.gz');components=component_records_fast(gt);aff,shape=load_nifti_geometry(DATA/f'images/{cid}_0000.nii.gz');geom=vessel_geometry_fast(DATA/f'vessel_masks/{cid}.nii.gz',shape,aff);rr=[];features=[]
  for comp in components:
   coords=comp['coords'];choices=[]
   for idx,score,lo,hi in selected:
    overlap=int(np.all((coords>=lo)&(coords<hi),axis=1).sum())
    if overlap/max(len(coords),1)>=.1:choices.append((score,idx,lo,hi,overlap))
   if not choices:continue
   score,idx,lo,hi,overlap=max(choices,key=lambda a:(a[0],-a[1]));glo=(coords.min(0)-.5).tolist();ghi=(coords.max(0)+.5).tolist();matches=[i for i,r in enumerate(records) if r['case_id']==cid and r['view']=='original' and r['sample_index']==0 and r['class_id']==comp['class_id'] and np.array_equal(r['low'],glo) and np.array_equal(r['high'],ghi)];assert len(matches)==1,(cid,comp['class_id'],matches)
   gi=matches[0];x=np.concatenate([compute_feature(geom,aff,lo,hi,'MR',False,vpair),multiscale(geom,aff,lo,hi)]).astype(np.float32);assert x.shape==(943,) and np.isfinite(x).all();features.append(x);gp=int(model.predict(old['X'][gi:gi+1])[0]);dp=int(model.predict(x[None])[0]);rr.append({'case_id':cid,'OOF_fold':owner[cid],'GT_row':gi,'label':int(comp['class_id']),'GT_box_prediction':gp,'OOF_box_prediction':dp,'candidate_index':idx,'score':score,'covered_GT_fraction':overlap/len(coords),'low':lo.tolist(),'high':hi.tolist()})
  np.savez_compressed(OUT/f'{cid}.npz',X=np.asarray(features,np.float32).reshape(-1,943));summary={'GT_components':len(components),'paired_components':len(rr),'selected_candidates':len(selected)};write_json(marker,{'rows':rr,'summary':summary});case_summary[cid]=summary;rows.extend(rr);print('E23_OOF_LOCATION_SOURCE',n,len(ids),cid,len(rr),flush=True)
 assignments={}
 for row in rows:assignments.setdefault((row['case_id'],row['candidate_index']),[]).append(row)
 paired=[];ambiguous=[]
 for key,items in assignments.items():
  if len({r['label'] for r in items})>1:ambiguous.extend(items);continue
  paired.append(max(items,key=lambda r:(r['covered_GT_fraction'],-r['GT_row'])))
 gc=sum(r['label']==r['GT_box_prediction'] for r in paired);dc=sum(r['label']==r['OOF_box_prediction'] for r in paired);write_json(OUT/'RESULT.json',{'n':len(paired),'GT_box_correct':gc,'OOF_box_correct':dc,'gap':gc-dc,'training_trigger_passed':len(paired)>=20 and gc-dc>=2,'rows':paired,'all_lesion_rows':rows,'excluded_multiclass_shared_candidate_rows':ambiguous,'matching_note':'For the training trigger, exclude multiclass shared-box ambiguities and count each candidate once; all underlying lesion rows remain available.','cases':case_summary,**config});print('E23_OOF_LOCATION_RESULT',len(paired),gc,dc,flush=True)
if __name__=='__main__':
 main()
 from scripts.astra6_e23.audit_oof_segmentation import main as audit_segmentation
 audit_segmentation()
