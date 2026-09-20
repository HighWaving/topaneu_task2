"""Same-case candidate distribution diagnosis, separate from F causal attribution."""
import json
import numpy as np
from scripts.astra6_e25.common import P,RUN
from scripts.astra6_e01.e01_common import load_boxes,box_to_native_bounds,select_candidates,sha256_file,write_json

def wide_indices(scores):
 # Preserve the frozen select_candidates tie order, so its operating top5 is
 # always a subset of the wider score>=.05/top20 source pool.
 return [int(i) for i in np.argsort(-np.asarray(scores)) if scores[i]>=.05][:20]

def old_source_candidates(cid,comps):
 path=P/('artifacts/fold1_source_center5_epoch60_20260909' if 'center5' in cid else 'artifacts/fold1_eval_center1_epoch60')/f'{cid}_boxes.pkl';boxes,scores,_=load_boxes(path);operating={i for i,_,_,_ in select_candidates(boxes,scores)};wide=wide_indices(scores);assert operating<=set(wide);rows=[]
 for i in wide:
  low,high=box_to_native_bounds(boxes[i]);counts=[int(np.all((c['coords']>=low)&(c['coords']<high),axis=1).sum()) for c in comps]
  rows.append({'candidate_index':i,'score':float(scores[i]),'operating_pool':i in operating,'components_any_overlap':[j for j,n in enumerate(counts) if n>0],'components_GT_coverage10pct':[j for j,(n,c) in enumerate(zip(counts,comps)) if n/len(c['coords'])>=.1]})
 return {'boxes_path':str(path),'boxes_sha256':sha256_file(path),'candidates':rows}

def summarize(ids):
 cases=[];lesions=[]
 def counts(rows,pool):
  rr=[r for r in rows if pool=='wide' or r['operating_pool']];pos=[r for r in rr if r['components_GT_coverage10pct']];neg=[r for r in rr if not r['components_any_overlap']];amb=[r for r in rr if r['components_any_overlap'] and not r['components_GT_coverage10pct']]
  return {'n':len(rr),'positive_candidates':len(pos),'negative_candidates':len(neg),'ambiguous_candidates':len(amb),'negative_score_quantiles10_50_90':np.percentile([r['score'] for r in neg],[10,50,90]).tolist() if neg else None}
 for cid in ids:
  m=json.loads((RUN/f'features/cases/{cid}.provenance.json').read_text());old=m['old_source_detector'];new=m['candidate_coverage'];row={'case_id':cid,'source':cid.split('_')[1],'GT_components':len(m['GT_lesions']),'negative_case':not m['GT_lesions'],'old_boxes_sha256':old['boxes_sha256'],'OOF_boxes_sha256':m['boxes_sha256'],'old':{},'OOF':{}}
  for pool in ['wide','operating']:
   row['old'][pool]=counts(old['candidates'],pool);row['OOF'][pool]=counts(new,pool)
   for g in m['GT_lesions']:
    j=g['component_index'];before=any(j in c['components_GT_coverage10pct'] for c in old['candidates'] if pool=='wide' or c['operating_pool']);after=any(j in c['components_GT_coverage10pct'] for c in new if pool=='wide' or c['operating_pool']);lesions.append({'case_id':cid,'source':row['source'],'pool':pool,**g,'old_covered10pct':before,'OOF_covered10pct':after})
  cases.append(row)
 summary={}
 for pool in ['wide','operating']:
  rr=[r for r in lesions if r['pool']==pool];summary[pool]={'old_covered_lesions':sum(r['old_covered10pct'] for r in rr),'OOF_covered_lesions':sum(r['OOF_covered10pct'] for r in rr),'GT_lesions':len(rr),'newly_uncovered':sum(r['old_covered10pct'] and not r['OOF_covered10pct'] for r in rr),'newly_covered':sum(not r['old_covered10pct'] and r['OOF_covered10pct'] for r in rr),'old_negative_candidates':sum(r['old'][pool]['negative_candidates'] for r in cases),'OOF_negative_candidates':sum(r['OOF'][pool]['negative_candidates'] for r in cases)}
 write_json(RUN/'evaluation/PAIRED_SOURCE_CANDIDATE_DISTRIBUTION.json',{'cases':cases,'lesions':lesions,'summary':summary,'no_model_fit_or_operating_threshold_change':True,'old_source_inventory_sha256':sha256_file(P/'artifacts/research_audit_20260909/OLD_SOURCE_CANDIDATE_INVENTORY.json'),'interpretation':'Same267source cases and current GT, frozen oldD versus completed OOF D, same operating score/topK rules. Counts/scores characterize candidate distribution, not adjudicated negative hardness or official FP counts. OldD trained268 historical cases; OOF D trains half267 with current data. Both use shared legacy GT-informed planning. TTA1 and raw cap200 match; old center1 batch4 differs from OOF batch1, while old center5 uses batch1. See OLD_SOURCE_CANDIDATE_INVENTORY.json; no byte-parity claim. Gradient exposure, training amount and data versions remain confounded, so this alone is not a causal OOF effect estimate.'})
