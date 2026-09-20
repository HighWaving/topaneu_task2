"""Official six and lesion-level paired C attribution, never refitting."""
import json
import numpy as np
from scripts.astra6_e28.common import P,RUN
from scripts.astra6_e01.e01_common import DATA,load_nifti,write_json,sha256_tree
from scripts.analysis.paired_segmentation_diagnostics import diagnose

def main():
 lock=json.loads((RUN/'PREDICTIONS_HASHED_BEFORE_EVAL_GT.json').read_text());assert lock['prediction_tree_sha256']==sha256_tree(RUN/'predictions');baseline=lock['baseline'];parent=P/('artifacts/astra6_e25_oof_shape_fp_filter_20260909' if baseline=='E25' else 'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909');old=json.loads((parent/'evaluation/lesion_diagnostics.json').read_text());cases={};rows=[]
 for cid,b in old['cases'].items():
  gt,aff,_=load_nifti(DATA/f'location_masks/{cid}.nii.gz');pred,pa,_=load_nifti(RUN/f'predictions/mr_center2_k05/{cid}.nii.gz');assert np.allclose(aff,pa,atol=1e-4);a=diagnose(gt,pred,aff);cases[cid]=a;assert len(a['lesions'])==len(b['lesions'])
  for i,(before,after) in enumerate(zip(b['lesions'],a['lesions'])):
   assert (before['class'],before['voxels'])==(after['class'],after['voxels']);rows.append({'case_id':cid,'lesion_index':i,'class':before['class'],'size_bin':before['size_bin'],'before':before,'after':after,'location_rescued':bool(after.get('class_correct',False) and not before.get('class_correct',False)),'location_lost':bool(before.get('class_correct',False) and not after.get('class_correct',False))})
  print('E28 lesion diagnosis',cid,flush=True)
 matched=[r for r in rows if r['after']['matched']];summary={'GT_components':len(rows),'matched':len(matched),'location_correct':sum(r['after']['class_correct'] for r in matched),'location_denominator':len(matched),'FP_per_case':sum(c['FP'] for c in cases.values())/len(cases),'matched_binary_Dice_mean':float(np.mean([r['after']['binary_dice'] for r in matched])) if matched else 0};write_json(RUN/'evaluation/lesion_diagnostics.json',{'E28':summary,'cases':cases,'paired_rows':rows,'location_rescued':sum(r['location_rescued'] for r in rows),'location_lost':sum(r['location_lost'] for r in rows),'baseline':baseline})
if __name__=='__main__':main()
