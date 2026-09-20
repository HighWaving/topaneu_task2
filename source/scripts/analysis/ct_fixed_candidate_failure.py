"""Diagnose the frozen CT5 candidate/filter boundary; never select a new threshold."""
from pathlib import Path
import json
import numpy as np
import nibabel as nib
from scipy.ndimage import label
from scripts.astra6_e01.e01_common import load_boxes,select_candidates,box_to_native_bounds,sha256_file,write_json,DATA
P=Path(__file__).resolve().parents[2]
R=P/'artifacts/astra6_e16_CT_expanded_segmentation_20260909'
def main():
 diagnosis=json.loads((R/'evaluation/lesion_diagnostics.json').read_text());cases={}
 for cid,dd in diagnosis['cases'].items():
  path=P/f'artifacts/ct_fold2_all109_boxes/{cid}_boxes.pkl';boxes,scores,_=load_boxes(path);selected={i for i,_,_,_ in select_candidates(boxes,scores)};saved=json.loads((R/f'predictions_ct/{cid}.nii.json').read_text());decisions={r['index']:r for r in saved['decisions']};im=nib.load(str(DATA/f'location_masks/{cid}.nii.gz'));gt=np.asanyarray(im.dataobj);rows=[]
  for cl in np.unique(gt):
   if cl==0:continue
   components,n=label(gt==cl)
   for ci in range(1,n+1):
    coords=np.argwhere(components==ci);hits=[]
    for rank,idx in enumerate(np.argsort(-scores),1):
     lo,hi=box_to_native_bounds(boxes[idx]);coverage=float(np.all((coords>=lo)&(coords<hi),axis=1).mean())
     if coverage>0:hits.append({'index':int(idx),'rank':rank,'score':float(scores[idx]),'GT_voxel_coverage':coverage,'operating':int(idx) in selected,'saved_filter_decision':decisions.get(int(idx))})
    operating=[h for h in hits if h['operating']];retained=[h for h in operating if h['saved_filter_decision'] and h['saved_filter_decision']['keep']]
    rows.append({'class':int(cl),'voxels':len(coords),'any_overlap_candidates':hits,'operating_max_GT_coverage':max([h['GT_voxel_coverage'] for h in operating],default=0),'retained_max_GT_coverage':max([h['GT_voxel_coverage'] for h in retained],default=0),'diagnostic_only_no_threshold_change':True})
  cases[cid]={'GT_components':rows,'saved_E16_diagnosis':dd['E16'],'boxes_sha256':sha256_file(path),'filter_records_sha256':sha256_file(R/f'predictions_ct/{cid}.nii.json')};print(cid,'done',flush=True)
 write_json(P/'artifacts/research_audit_20260909/CT5_FIXED_CANDIDATE_FAILURE.json',{'cases':cases,'interpretation':'Coverage is a diagnostic of saved native boxes, not official lesion matching or a proposed operating threshold. CT5 is repeatedly observed, has only5 positive cases, and is not a representative independent generalization estimate. No model inference, fitting, threshold selection or GT vessel input performed.'})
if __name__=='__main__':main()
