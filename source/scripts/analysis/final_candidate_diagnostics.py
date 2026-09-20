"""Separate detector/FP-filter box coverage from final mask lesion matching."""
import json
import numpy as np
from scripts.astra6_e01.e01_common import P,DATA,load_boxes,select_candidates,load_nifti,write_json
from scripts.astra6_e03.run_e03 import component_records_fast

def main():
 mr=P/'artifacts/astra6_e14_MR_filter_seed_stability_20260909';ct=P/'artifacts/astra6_e16_CT_expanded_segmentation_20260909';mrrows=[json.loads(s) for s in (mr/'candidate_predictions.jsonl').read_text().splitlines()];mrkeep={(r['case_id'],r['original_index']):r['filter_keep'] for r in mrrows};out={}
 for mod,root in [('MR',mr),('CT',ct)]:
  ids=json.loads((P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z/eval_case_ids.json').read_text()) if mod=='MR' else json.loads((ct/'source_split.json').read_text())['fixed_CT5'];cases={};ledger=[];counts={'before_F_candidates':0,'after_F_candidates':0,'GT_components':0,'before_F_any_voxel_hit':0,'after_F_any_voxel_hit':0,'before_F_10percent_hit':0,'after_F_10percent_hit':0,'before_F_raster_box_hit':0,'after_F_raster_box_hit':0}
  for cid in ids:
   gt,aff,shape=load_nifti(DATA/f'location_masks/{cid}.nii.gz');comps=component_records_fast(gt);folder=P/('artifacts/fold1_eval_center2_epoch60' if mod=='MR' else 'artifacts/ct_fold2_all109_boxes');bx,sc,_=load_boxes(folder/f'{cid}_boxes.pkl');selected=select_candidates(bx,sc)
   keep={i:mrkeep[cid,i] for i,_,_,_ in selected} if mod=='MR' else {r['index']:r['keep'] for r in json.loads((ct/f'predictions_ct/{cid}.nii.json').read_text())['decisions']};after=[r for r in selected if keep[r[0]]];counts['before_F_candidates']+=len(selected);counts['after_F_candidates']+=len(after);counts['GT_components']+=len(comps)
   for c in comps:
    rec={'case_id':cid,'class':c['class_id'],'component_id':c['component_id'],'voxels':len(c['coords']),'diameter_mm':float((6*len(c['coords'])*abs(np.linalg.det(aff[:3,:3]))/np.pi)**(1/3))}
    for stage,candidates in [('before_F',selected),('after_F',after)]:
     fractions=[float(np.mean(np.all((c['coords']>=lo)&(c['coords']<hi),axis=1))) for _,_,lo,hi in candidates];raster=[bool(np.any(np.all((c['coords']>=np.floor(lo))&(c['coords']<np.ceil(hi)),axis=1))) for _,_,lo,hi in candidates];coverage=max(fractions,default=0);rec[stage+'_max_GT_voxel_fraction']=coverage;rec[stage+'_raster_box_hit']=any(raster);counts[stage+'_any_voxel_hit']+=int(coverage>0);counts[stage+'_10percent_hit']+=int(coverage>=.1);counts[stage+'_raster_box_hit']+=int(any(raster))
    ledger.append(rec)
   cases[cid]={'before_F':len(selected),'after_F':len(after)};print(mod,cid,flush=True)
  counts['cases']=len(ids);out[mod]={'summary':counts,'cases':cases,'lesions':ledger}
 write_json(P/'reports/FINAL_CANDIDATE_DIAGNOSTICS_20260909.json',{'definitions':{'any_voxel_hit':'At least one GT voxel center lies inside the continuous predicted box','10percent_hit':'At least10percent of a GT component voxel centers are inside a selected box, matching F training label convention','raster_box_hit':'Any GT voxel in floor(low):ceil(high), a raster crop coverage bound rather than a segmentation result','post_filter':'Frozen best F decisions; no thresholds fitted here'},'versions':out});print({k:v['summary'] for k,v in out.items()},flush=True)
if __name__=='__main__':main()
