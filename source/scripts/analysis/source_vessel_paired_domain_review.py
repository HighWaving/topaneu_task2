"""Same-case organizer-vessel / TA36 diagnostic using existing source artifacts only."""
import json
from pathlib import Path
import nibabel as nib,numpy as np
from scripts.astra6_e01.e01_common import P,DATA,write_json,sha256_file
R=P/'artifacts/review_current_model_20260910';CACHE=P/'artifacts/source_MR_TA36_distribution_audit_20260909'
def main():
 dest=R/'SOURCE_PAIRED_VESSEL_DOMAIN.json'
 if dest.exists():return
 rows=[];negative=[];inputs=[]
 for file in sorted(CACHE.glob('*/predicted_vessel.nii.gz')):
  cid=file.parent.name;prov=json.loads((file.parent/'PROVENANCE.json').read_text());assert prov['GT_not_input'];assert sha256_file(DATA/f'images/{cid}_0000.nii.gz')==prov['image_sha256'];g=nib.load(str(DATA/f'location_masks/{cid}.nii.gz'));v=nib.load(str(DATA/f'vessel_masks/{cid}.nii.gz'));t=nib.load(str(file));assert g.shape==v.shape==t.shape and np.allclose(g.affine,v.affine,atol=1e-4) and np.allclose(g.affine,t.affine,atol=1e-4);gt=np.asanyarray(g.dataobj)>0;inputs.append(cid)
  if not gt.any():negative.append(cid);continue
  a=np.asanyarray(v.dataobj)[gt]>0;b=np.asanyarray(t.dataobj)[gt]>0;rows.append({'case_id':cid,'GT_voxels':int(gt.sum()),'organizer_vessel_GT_overlap':float(a.mean()),'TA36_GT_overlap':float(b.mean()),'delta':float(b.mean()-a.mean()),'TA36_sha256':sha256_file(file)});print('PAIRED SOURCE VESSEL',cid,flush=True)
 delta=np.array([r['delta'] for r in rows]);rng=np.random.default_rng(20260910);boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(5000)]);write_json(dest,{'n_cached_cases':len(inputs),'n_positive_cases':len(rows),'negative_cases':negative,'organizer_mean':float(np.mean([r['organizer_vessel_GT_overlap'] for r in rows])),'TA36_mean':float(np.mean([r['TA36_GT_overlap'] for r in rows])),'paired_delta_mean':float(delta.mean()),'paired_case_bootstrap_95CI':np.percentile(boot,[2.5,97.5]).tolist(),'rows':rows,'no_new_model_or_inference':True,'interpretation':'Same-case difference removes between-cohort case/scanner confounding from the earlier E31 diagnostic. It does not establish clinical vessel correctness, upstream training independence, or causal S reliance. GT used only for offline overlap diagnosis.','cached_MR_source_predictions_exist':True});print('PAIRED SOURCE VESSEL COMPLETE',len(rows),float(delta.mean()),flush=True)
if __name__=='__main__':main()
