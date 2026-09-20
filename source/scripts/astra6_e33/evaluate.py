"""D proposal coverage is reported separately from final native lesion performance."""
import json,math
import numpy as np
from scripts.astra6_e01.e01_common import P,DATA,load_nifti,load_boxes,select_candidates,write_json,sha256_tree
from scripts.astra6_e04.run_e04 import component_records_fast
RUN=P/'artifacts/astra6_e33_source_only_detector_20260910'
def main():
 import scripts.astra6_e28.evaluate as d
 d.RUN=RUN;dest=RUN/'evaluation/lesion_diagnostics.json'
 if not dest.exists():d.main()
 result=json.loads(dest.read_text())
 if 'E28' in result:result['E33']=result.pop('E28')
 write_json(dest,result);lock=json.loads((RUN/'PREDICTIONS_HASHED_BEFORE_EVAL_GT.json').read_text());assert sha256_tree(RUN/'predictions')==lock['prediction_tree_sha256'];cases=json.loads((RUN/'DETECTOR_INFERENCE_COMPLETE.json').read_text())['cases'];rows=[];background=[]
 for cid in cases:
  gt,aff,_=load_nifti(DATA/f'location_masks/{cid}.nii.gz');components=component_records_fast(gt);voxelvol=abs(np.linalg.det(aff[:3,:3]));pools={}
  for arm,path in [('E17',P/f'artifacts/fold1_eval_center2_epoch60/{cid}_boxes.pkl'),('E33',RUN/f'comparison_boxes/{cid}_boxes.pkl')]:
   bx,sc,_=load_boxes(path);pools[arm]=select_candidates(bx,sc);candidates=[]
   for ix,score,lo,hi in pools[arm]:
    coverage=[float(np.all((c['coords']>=lo)&(c['coords']<hi),axis=1).mean()) for c in components];candidates.append({'index':ix,'score':score,'max_GT_coverage':max(coverage,default=0),'best_GT_index':int(np.argmax(coverage)) if coverage else None})
   background.append({'case_id':cid,'arm':arm,'candidate_count':len(candidates),'zero_GT_overlap_candidates':sum(c['max_GT_coverage']==0 for c in candidates),'candidates':candidates})
  for i,c in enumerate(components):
   n=len(c['coords']);diam=(6*n*voxelvol/math.pi)**(1/3);r={'case_id':cid,'component_index':i,'class':c['class_id'],'diameter_mm':diam,'size_bin':'<=3' if diam<=3 else '(3,5]' if diam<=5 else '(5,7]' if diam<=7 else '>7','GT_voxels':n}
   for arm in pools:
    coverage=[float(np.all((c['coords']>=lo)&(c['coords']<hi),axis=1).mean()) for ix,score,lo,hi in pools[arm]];r[arm]={'max_GT_coverage':max(coverage,default=0),'candidate_cover_10percent':max(coverage,default=0)>=.1,'candidate_cover_90percent':max(coverage,default=0)>=.9}
   rows.append(r)
  print('E33 D DIAGNOSIS',cid,flush=True)
 summary={arm:{'GT_components':len(rows),'coverage10_count':sum(r[arm]['candidate_cover_10percent'] for r in rows),'coverage90_count':sum(r[arm]['candidate_cover_90percent'] for r in rows),'mean_best_GT_coverage':float(np.mean([r[arm]['max_GT_coverage'] for r in rows])),'raw_zero_overlap_candidates_per_case':sum(r['zero_GT_overlap_candidates'] for r in background if r['arm']==arm)/40} for arm in pools}
 write_json(RUN/'evaluation/DETECTOR_ATTRIBUTION.json',{'comparison':'Same .3/top5 candidate policy, fixed downstream weights; changed D training/plan package may alter downstream feature distribution.','candidate_background_is_not_final_FP_per_case':True,'summaries':summary,'paired_GT_rows':rows,'case_candidates':background,'limitations':'All MR40 cases positive; repeated development; source-only D planning removes a known exposure, not all pipeline/history uncertainty.'})
if __name__=='__main__':main()
