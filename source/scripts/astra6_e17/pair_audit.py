"""Compare the same source lesions, using each lesion's highest-scoring detector box."""
import json,shutil
import numpy as np
from scripts.astra6_e17.prepare import RUN,PARENT,DATA,component_records_fast,load_nifti,sha256_file,write_json

def main():
 rec=[json.loads(s) for s in (RUN/'features/train_records.jsonl').read_text().splitlines()];split=json.loads((RUN/'source_split.json').read_text());path=RUN/'evaluation/SOURCE_PRECONDITION.json';original=RUN/'evaluation/SOURCE_UNPAIRED_DESCRIPTIVE.json'
 if not original.exists():shutil.copy2(path,original)
 old=json.loads(original.read_text());gt_scores=dict(zip(split.get('unpaired_development_GT_rows',split['development_GT_rows']),old['baseline']['GT_boxes']['scores']));det_scores=dict(zip(split.get('development_detector_rows_all',split['development_detector_rows']),old['baseline']['detector_boxes']['scores']));lookup={}
 for i,r in enumerate(rec[:4986]):
  if r['view']=='original' and r['sample_index']==0:lookup.setdefault((r['case_id'],r['source_class_id']),[]).append(i)
 component_cache={}
 for i,r in enumerate(rec[4986:],4986):
  candidates=lookup[(r['case_id'],r['source_class_id'])]
  if len(candidates)==1:match=candidates[0]
  else:
   cid=r['case_id']
   if cid not in component_cache:gt,_,_=load_nifti(DATA/f'location_masks/{cid}.nii.gz');component_cache[cid]=component_records_fast(gt)
   comp=next(c for c in component_cache[cid] if c['class_id']==r['source_class_id'] and c['component_id']==r['component_id']);lo=comp['coords'].min(0)-.5;hi=comp['coords'].max(0)+.5;match=next(j for j in candidates if np.array_equal(rec[j]['low'],lo) and np.array_equal(rec[j]['high'],hi))
  r['matched_GT_base_row']=match
 all_dv=split.get('development_detector_rows_all',split['development_detector_rows']);primary={}
 for i in all_dv:
  match=rec[i]['matched_GT_base_row']
  if match not in primary or rec[i]['score']>rec[primary[match]]['score']:primary[match]=i
 gt_rows=sorted(primary);det_rows=[primary[i] for i in gt_rows];g=np.array([gt_scores[i] for i in gt_rows]);d=np.array([det_scores[i] for i in det_rows]);gap=float((g-d).mean());assert len(g)==len(d) and len(g)>20
 split['unpaired_development_GT_rows']=split.get('unpaired_development_GT_rows',split['development_GT_rows']);split['development_detector_rows_all']=all_dv;split['development_GT_rows']=gt_rows;split['development_detector_rows']=det_rows;write_json(RUN/'source_split.json',split);(RUN/'features/train_records.jsonl').write_text('\n'.join(json.dumps(r) for r in rec)+'\n');ready=json.loads((RUN/'features/SOURCE_READY.json').read_text());ready['records_sha256']=sha256_file(RUN/'features/train_records.jsonl');write_json(RUN/'features/SOURCE_READY.json',ready)
 report={'baseline':{'GT_boxes':{'n':len(g),'crop_Dice_mean':float(g.mean()),'scores':g.tolist()},'detector_boxes':{'n':len(d),'crop_Dice_mean':float(d.mean()),'scores':d.tolist()}},'paired_same_source_lesions':True,'selection':'highest detector score per GT component, matched by class and exact original GT bounds when component IDs ambiguous','paired_GT_rows':gt_rows,'paired_detector_rows':det_rows,'source_GT_dev_components_without_selected_detector_match':len(gt_scores)-len(g),'Dice_gap':gap,'proceed_to_training':gap>=.02,'source_lesions_with_detector_crop_degradation':int((d<g).sum()),'no_MR40_access':True};write_json(path,report);print('PAIRED_SOURCE_PRECONDITION',len(g),gap,gap>=.02,flush=True)
if __name__=='__main__':main()
