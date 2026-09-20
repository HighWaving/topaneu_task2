"""Read-only native GT/predicted-vessel overlap to check an S31 input-domain hypothesis."""
import json,time
from pathlib import Path
import nibabel as nib,numpy as np
from scripts.astra6_e31.common import P,RUN
from scripts.astra6_e01.e01_common import DATA,TA36_DIR,write_json

def main():
 dest=RUN/'VESSEL_INPUT_DOMAIN_DIAGNOSTIC.json'
 if dest.exists():return
 records=[json.loads(s) for s in (RUN/'features/records.jsonl').read_text().splitlines()];split=json.loads((RUN/'source_split.json').read_text());source=sorted({records[i]['case_id'] for i in split['validation_rows']});mr40=json.loads((P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z/eval_case_ids.json').read_text());rows=[]
 for cohort,ids in [('source_selection',source),('MR40_development',mr40)]:
  for cid in ids:
   gp=DATA/f'location_masks/{cid}.nii.gz';vp=(DATA/'vessel_masks' if cohort=='source_selection' else TA36_DIR)/f'{cid}.nii.gz';g=nib.load(str(gp));v=nib.load(str(vp));assert g.shape==v.shape and np.allclose(g.affine,v.affine,atol=1e-4);gt=np.asanyarray(g.dataobj);ves=np.asanyarray(v.dataobj);target=gt>0;values=ves[target];assert len(values)>0;labels,counts=np.unique(values,return_counts=True);rows.append({'cohort':cohort,'case_id':cid,'GT_voxels':len(values),'GT_volume_mm3':float(len(values)*abs(np.linalg.det(g.affine[:3,:3]))),'GT_fraction_inside_predicted_vessel':float((values>0).mean()),'vessel_labels_on_GT':dict(zip(map(str,labels.tolist()),counts.tolist()))});print('E31 vessel domain',cohort,cid,flush=True)
 summary={}
 for cohort in ['source_selection','MR40_development']:
  z=[r for r in rows if r['cohort']==cohort];v=[r['GT_fraction_inside_predicted_vessel'] for r in z];summary[cohort]={'n_cases':len(z),'case_mean_GT_vessel_overlap':float(np.mean(v)),'case_median_GT_vessel_overlap':float(np.median(v)),'p10_p90':np.percentile(v,[10,90]).tolist()}
 write_json(dest,{'summary':summary,'cases':rows,'interpretation':'GT used only for offline diagnostic. Source organizer prediction vs deployment TA36 plus scanner/anatomy/case distributions are confounded; overlap alone does not prove causal model dependence or vessel correctness.'});print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
