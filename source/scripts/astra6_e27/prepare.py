"""Append branch-origin descriptors to the frozen E02 source bank."""
from pathlib import Path
import json,time,shutil
import numpy as np
from scripts.astra6_e27.common import *
from scripts.astra6_e01.e01_common import DATA,load_nifti_geometry,write_json,sha256_file
from scripts.delivery.geometry import vessel_geometry_fast
BASE=P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z';TA=P/'artifacts/source_MR_TA36_distribution_audit_20260909'
def main():
 for d in ['features/cases','features/source_TA','model','evaluation','logs','predictions/mr_center2_k05']:(RUN/d).mkdir(parents=True,exist_ok=True)
 schema=json.loads((BASE/'feature_schema.json').read_text());assert {int(k):int(v) for k,v in schema['vessel_lr_pair'].items()}==VPAIR;write_json(RUN/'FEATURE_PREFLIGHT.json',synthetic_geometry_check())
 config={'experiment':'E27','hypothesis':'Distances to whole artery labels do not identify branch origins; junction-anchor offsets and along-span position may distinguish named junctions from neighboring arterial segments.','new_features':292,'total_features':1235,'anchor_rule':'64 nearest child-to-parent points including equal-distance ties, inverse-square distance weights; noGT or learned geometric threshold','pairs':PAIRS,'paths':PATHS,'source':str(BASE),'classifier':'same ExtraTrees512,depth16,minleaf2,maxfeatures.5; developmentseed20260909 and finalseed20260905 match respectiveE05/E02 comparators','CPU_jobs':1,'budget_hours':2,'checkpoint':'sourcecase feature/provenance cache; forest checkpoints every64trees through512; fullsource fit alwayscompleted','source_gate':'GT56 correct>=40 and actualTA36 detector31 correct>=27, OR GT56correct>=43 and TA31correct>=25. Only source-heldout classifiers used.','source_anatomy':'Organizer-provided predicted silver for fitting; actual frozenTA36 predictions on29heldoutMRcases/31lesions additionallygate transfer','no_MR40_CT5_fit':True,'holdout_comparison':'Only if sourcegatepasses, oneMR40 C-only inference and official6metrics with E17S/E14F frozen; no newheldout errors fed back into training','no_all_stage_LOCO_claim':'Prior TA36 patient-list limitations remain disclosed.'};write_json(RUN/'config.json',config)
 records=[json.loads(s) for s in (BASE/'features/train_records.jsonl').read_text().splitlines()];cases=sorted({r['case_id'] for r in records});assert len(cases)==218 and not any('center2' in c for c in cases);extra=np.empty((len(records),292),np.float32);timings={}
 for n,cid in enumerate(cases,1):
  ix=[i for i,r in enumerate(records) if r['case_id']==cid];dest=RUN/f'features/cases/{cid}.npz'
  if dest.exists():extra[ix]=np.load(dest)['X'];continue
  t=time.monotonic();aff,shape=load_nifti_geometry(DATA/f'images/{cid}_0000.nii.gz');geom=vessel_geometry_fast(DATA/f'vessel_masks/{cid}.nii.gz',shape,aff);tt=time.monotonic();j=anchors(geom);anchor_seconds=time.monotonic()-tt;rows=[]
  for i in ix:
   r=records[i];x=features(j,aff,r['low'],r['high']);rows.append(mirrored(x) if r['view']=='mirror' else x)
  xx=np.asarray(rows,np.float32);extra[ix]=xx;np.savez_compressed(dest,X=xx);timings[cid]={'seconds':time.monotonic()-t,'anchor_seconds':anchor_seconds,'missing_pairs':sum(v is None for v in j.values())};write_json(dest.with_suffix('.json'),timings[cid]);print('E27_SOURCE_FEATURES',n,len(cases),cid,timings[cid],flush=True)
 z=np.load(BASE/'features/train.npz');X=np.concatenate([z['X'],extra],axis=1);assert X.shape==(4986,1235) and np.isfinite(X).all();np.savez_compressed(RUN/'features/train.npz',X=X,y=z['y'],sample_weight=z['sample_weight']);shutil.copy2(BASE/'features/train_records.jsonl',RUN/'features/train_records.jsonl')
 audit=json.loads((TA/'RESULT.json').read_text());assert audit['n']==31 and audit['TA36_correct']==25;rows=audit['rows'];xx=[]
 for cid in sorted({r['case_id'] for r in rows}):
  rr=[r for r in rows if r['case_id']==cid];case=TA/cid;dest=RUN/f'features/source_TA/{cid}.npz'
  if dest.exists():xx.extend(np.load(dest)['X']);continue
  aff,shape=load_nifti_geometry(DATA/f'images/{cid}_0000.nii.gz');geom=vessel_geometry_fast(case/'predicted_vessel.nii.gz',shape,aff);j=anchors(geom);old=np.load(case/'features.npz')['X'];assert len(old)==len(rr);detrec=[json.loads(s) for s in (P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909/features/train_records.jsonl').read_text().splitlines()];new=[]
  for x,r in zip(old,rr):
   bounds=detrec[r['detector_row']];new.append(np.concatenate([x,features(j,aff,bounds['low'],bounds['high'])]))
  new=np.asarray(new,np.float32);np.savez_compressed(dest,X=new);xx.extend(new);write_json(dest.with_suffix('.json'),{'TA36_prediction_sha256':sha256_file(case/'predicted_vessel.nii.gz'),'upstream_provenance':json.loads((case/'PROVENANCE.json').read_text())});print('E27_SOURCE_TA_FEATURES',cid,flush=True)
 assert rows==sorted(rows,key=lambda r:r['case_id']);np.savez_compressed(RUN/'features/source_TA.npz',X=np.asarray(xx,np.float32),y=[r['label'] for r in rows]);write_json(RUN/'features/source_TA_rows.json',rows);write_json(RUN/'features/READY.json',{'n_rows':len(records),'n_cases':218,'n_features':1235,'train_sha256':sha256_file(RUN/'features/train.npz'),'records_sha256':sha256_file(RUN/'features/train_records.jsonl'),'TA31_sha256':sha256_file(RUN/'features/source_TA.npz'),'feature_code_sha256':sha256_file(Path(__file__).with_name('common.py')),'anatomy_inputs_are_predicted_vessels':True,'source_box_disclosure':'GT-derived boxes are used in supervised source fitting; sourceTA31 uses actualdetector boxes. Deployed inference uses noGT.'});print('E27_SOURCE_READY',flush=True)
if __name__=='__main__':main()
