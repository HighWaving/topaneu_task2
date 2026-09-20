import json
from pathlib import Path
import numpy as np
from scripts.astra6_e01.e01_common import P,load_boxes,select_candidates,write_json
out={}
for name,folder in [('E06','astra6_e06_image_fp_filter_20260909'),('E08','astra6_e08_mr_only_fp_filter_20260909')]:
 r=P/'artifacts'/folder;records=[json.loads(s) for s in (r/'features/records.jsonl').read_text().splitlines()];p=json.loads((r/'model/development_best_predictions.json').read_text());cache={};items=[]
 for idx,y,prob in zip(p['rows'],p['y'],p['p']):
  row=records[idx];cid=row['case_id'];mod='MR' if '_mr_' in cid else 'CT'
  if cid not in cache:
   root=P/('artifacts/fold1_eval_center1_epoch60' if mod=='MR' else 'artifacts/ct_fold2_all109_boxes');bx,sc,_=load_boxes(root/f'{cid}_boxes.pkl');cache[cid]={x[0] for x in select_candidates(bx,sc)}
  items.append({'case_id':cid,'modality':mod,'y':y,'p':prob,'source_detector_score':row['score'],'deployed_pool':row['original_index'] in cache[cid]})
 result={}
 for mod in ['MR','CT']:
  rows=[z for z in items if z['modality']==mod]
  if not rows:continue
  positives=[z for z in rows if z['y']];dep=[z for z in rows if z['deployed_pool']];pos=[z for z in dep if z['y']];neg=[z for z in dep if not z['y']]
  result[mod]={'all_pool_positive':len(positives),'deployed_pool_positive':len(pos),'deployed_pool_negative':len(neg),'minimum_all_pool_positive':min(positives,key=lambda z:z['p']),'minimum_deployed_pool_positive':min(pos,key=lambda z:z['p']) if pos else None}
 out[name]=result
write_json(P/'reports/SOURCE_FILTER_CALIBRATION_AUDIT_20260909.json',{'policy':'diagnostic only; existing thresholds not changed; all labels/probabilities source-validation only','models':out});print(json.dumps(out,indent=2))
