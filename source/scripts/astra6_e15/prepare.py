"""Align CT location-classifier source vessel inputs with actual inference TA36 outputs."""
from pathlib import Path
import json,shutil
import numpy as np
from scripts.astra6_e01.e01_common import P,DATA,load_nifti_geometry,compute_feature,sha256_file,write_json
from scripts.astra6_e02.run_e02 import multiscale,extended_schema,E01
from scripts.delivery.geometry import vessel_geometry_fast
RUN=P/'artifacts/astra6_e15_CT_predicted_vessel_location_20260909';PARENT=P/'artifacts/astra6_e09_CT_expanded_location_20260909'
def main():
 for s in ['features/cases','model','evaluation','predictions_ct']:(RUN/s).mkdir(parents=True,exist_ok=True)
 config={'experiment':'E15','hypothesis':'CT location training on the same TA36 predicted-vessel distribution used at inference improves branch assignment versus organizer-provided source silver masks','change':'replace only104CT source vessel feature rows in E09; preserve all MR source rows, labels, candidate boxes, augmentation, weights and ExtraTrees512 parameters','source_validation':'same8CT2 source development cases, excluded from both E09 development and E15 development fit; both scored on actualTA36 features','control':'best CT E09C/E04S/E11F/E13anatomicalF','source':'260positivecases, including104CT outsideCT5; actualTA36 already available; no additional data labels','budget_hours':3,'checkpoint_every_trees':64,'adoption':'fixedCT5 all-six Pareto improvement versusCT_E13; no foreground/matched-lesion change','inference_CT_only':True,'no_CT5_or_MR40_fit':True}
 write_json(RUN/'config.json',config);shutil.copy2(PARENT/'source_split.json',RUN/'source_split.json');shutil.copy2(PARENT/'features/train_records.jsonl',RUN/'features/train_records.jsonl');rec=[json.loads(s) for s in (RUN/'features/train_records.jsonl').read_text().splitlines()];old=np.load(PARENT/'features/train.npz');X=old['X'].copy();schema=json.loads((E01/'feature_schema.json').read_text());_,pair,perm=extended_schema(schema);groups={}
 for i,r in enumerate(rec):
  if '_ct_' in r['case_id']:groups.setdefault(r['case_id'],[]).append(i)
 assert len(groups)==104;assert not set(groups)&set(json.loads((RUN/'source_split.json').read_text())['fixed_CT5']);hashes={}
 for n,(cid,ix) in enumerate(sorted(groups.items()),1):
  vp=P/f'artifacts/ta36_ct_output/{cid}.nii.gz';hashes[cid]=sha256_file(vp);dest=RUN/f'features/cases/{cid}.npz'
  if dest.exists():a=np.load(dest);assert np.array_equal(a['indices'],ix);X[ix]=a['X'];continue
  aff,shape=load_nifti_geometry(DATA/f'images/{cid}_0000.nii.gz');geom=vessel_geometry_fast(vp,shape,aff);xx=[]
  for i in ix:
   r=rec[i];lo=np.array(r['low']);hi=np.array(r['high']);mirror=r['view']=='mirror';extra=multiscale(geom,aff,lo,hi);xx.append(np.concatenate([compute_feature(geom,aff,lo,hi,'CT',mirror,pair),extra[perm] if mirror else extra]))
  xx=np.asarray(xx,np.float32);X[ix]=xx;np.savez_compressed(dest,X=xx,indices=ix);print('CT_TA36_location_features',n,104,cid,flush=True)
 mr=[i for i,r in enumerate(rec) if '_mr_' in r['case_id']];assert np.array_equal(X[mr],old['X'][mr]) and np.isfinite(X).all();np.savez_compressed(RUN/'features/train.npz',X=X,y=old['y']);write_json(RUN/'features/READY.json',{'rows':len(rec),'feature_dim':943,'source_positive_cases':260,'CT_source_cases':104,'MR_features_bit_identical':True,'TA36_source_prediction_sha256':hashes,'features_sha256':sha256_file(RUN/'features/train.npz'),'records_sha256':sha256_file(RUN/'features/train_records.jsonl'),'parent_features_sha256':sha256_file(PARENT/'features/train.npz')});print('SOURCE_READY',flush=True)
if __name__=='__main__':main()
