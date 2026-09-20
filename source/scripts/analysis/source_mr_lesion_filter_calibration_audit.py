"""Check whether duplicate positives unnecessarily constrain the frozen MR F gate."""
import json
import numpy as np
from scripts.astra6_e01.e01_common import P,DATA,load_nifti,write_json,sha256_file
from scripts.astra6_e04.run_e04 import component_records_fast

def main():
 out=P/'artifacts/source_MR_lesion_filter_calibration_audit_20260909';out.mkdir(exist_ok=True);r=P/'artifacts/astra6_e19_MR_anatomical_fp_filter_20260909';split=json.loads((r/'source_split.json').read_text());records=[json.loads(s) for s in (r/'features/records.jsonl').read_text().splitlines()];prob=np.load(r/'features/development_image_probability.npy');indices=split['anatomical_calibration_rows'];old=split['parent_image_threshold'];assert len(indices)==52 and sum(records[i]['y'] for i in indices)==43
 write_json(out/'config.json',{'hypothesis':'Candidate-level minimum-positive threshold may preserve redundant proposals; source lesion coverage may permit a higher image-only filter threshold','calibration':'Highest threshold preserving every baseline-covered source GT component (>=10percent voxel coverage), computed from maximum image probability among candidates covering each component','gate':'Retain all baseline source components and remove at least2additional y0 candidates; thresholds frozen before any MR40 inference','source_only':True,'no_new_model_training':True,'baseline_parent_threshold':old,'no_MR40_or_CT5_access':True})
 groups={};lesions=[]
 for i in indices:assert records[i]['development'];groups.setdefault(records[i]['case_id'],[]).append(i)
 for cid,ix in sorted(groups.items()):
  gt,aff,shape=load_nifti(DATA/f'location_masks/{cid}.nii.gz')
  for comp in component_records_fast(gt):
   points=comp['coords'];covers=[]
   for i in ix:
    v=records[i];fraction=float(np.mean(np.all((points>=v['low'])&(points<v['high']),axis=1)))
    if fraction>=.1:covers.append({'row':i,'image_probability':float(prob[i]),'GT_fraction_in_box':fraction,'training_y':v['y']})
   if covers:lesions.append({'case_id':cid,'class_id':comp['class_id'],'component_id':comp['component_id'],'GT_voxels':len(points),'covering_candidates':covers,'best_image_probability':max(v['image_probability'] for v in covers)})
  print('SOURCE_LESION_CALIBRATION',cid,flush=True)
 threshold=float(np.nextafter(min(a['best_image_probability'] for a in lesions),0.));assert threshold>=old-1e-7;baseline_negative=[i for i in indices if not records[i]['y']];removed=[i for i in baseline_negative if prob[i]<threshold];lost=sum(not any(a['image_probability']>=threshold for a in r['covering_candidates']) for r in lesions);assert lost==0;result={'source_cases':len(groups),'baseline_covered_components':len(lesions),'candidate_positive_count':43,'components_with_multiple_candidates':sum(len(r['covering_candidates'])>1 for r in lesions),'old_threshold':old,'lesion_preserving_threshold':threshold,'source_negatives_before':len(baseline_negative),'additional_negatives_removed':len(removed),'removed_negative_rows':removed,'source_components_lost':lost,'gate_passed':len(removed)>=2,'source_only':True,'probability_sha256':sha256_file(r/'features/development_image_probability.npy'),'lesions':lesions};write_json(out/'RESULT.json',result);print('LESION_CALIBRATION_RESULT',{k:v for k,v in result.items() if k!='lesions'},flush=True)
if __name__=='__main__':main()
