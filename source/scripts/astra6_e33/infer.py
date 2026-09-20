"""D-only contrast with the explicit-file retained C02/S17/F14 refinement."""
import json,subprocess,sys,time
from pathlib import Path
from scripts.astra6_e01.e01_common import P,DATA,TA36_DIR,load_boxes,select_candidates,sha256_file,sha256_tree,write_json
RUN=P/'artifacts/astra6_e33_source_only_detector_20260910';BASE=P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z';F=P/'artifacts/astra6_e14_MR_filter_seed_stability_20260909/model';S=P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909/model'
def main():
 if (RUN/'PREDICTIONS_HASHED_BEFORE_EVAL_GT.json').exists():return
 lock=json.loads((RUN/'DETECTOR_INFERENCE_COMPLETE.json').read_text());fr=json.loads((F/'LOCKED.json').read_text());assert sha256_file(F/'THRESHOLD.json')==fr['threshold_sha256'];threshold=json.loads((F/'THRESHOLD.json').read_text())['threshold'];rows=[];proof={};start=time.monotonic()
 for cid in lock['cases']:
  boxes=RUN/f'comparison_boxes/{cid}_boxes.pkl';assert sha256_file(boxes)==lock['boxes_sha256'][cid];out=RUN/f'predictions/mr_center2_k05/{cid}.nii.gz';ledger=out.with_suffix('.json')
  if not out.exists() or not ledger.exists():
   subprocess.run([sys.executable,'-u','-m','scripts.delivery.refine_with_filter','--image',str(DATA/f'images/{cid}_0000.nii.gz'),'--predicted-vessel',str(TA36_DIR/f'{cid}.nii.gz'),'--boxes',str(boxes),'--classifier',str(BASE/'model/classifier.joblib'),'--segmentation',str(S/'final_last.pt'),'--modality','MR','--device','cuda:0','--fp-filter',str(F/'final_last.pt'),'--fp-threshold',str(threshold),'--output',str(out)],cwd=P,check=True)
  report=json.loads(ledger.read_text());decisions={r['index']:r for r in report['filter_decisions']};kept={r['index']:r for r in report['candidates']};bx,sc,_=load_boxes(boxes)
  for idx,score,lo,hi in select_candidates(bx,sc):
   q=decisions[idx];r={'case_id':cid,'original_index':idx,'score':score,'low':lo.tolist(),'high':hi.tolist(),'filter_probability':q['probability'],'filter_keep':q['keep']}
   if q['keep']:r.update(predicted_class_id=kept[idx]['class'],empty_fallback=kept[idx]['ellipsoid_empty_fallback'])
   rows.append(r)
  proof[cid]={'new_D_sha256':lock['checkpoint_sha256'],'C02_S17_F14_weights_unchanged':True,'legacy_empty_S_ellipse_count':sum(r['ellipsoid_empty_fallback'] for r in kept.values()),'seconds':report['total_seconds'],'peak_rss_kb':report['peak_rss_kb']};print('E33 REFINEMENT',cid,flush=True)
 (RUN/'candidate_predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n');write_json(RUN/'FIXED_DOWNSTREAM.json',{'cases':proof,'C_sha256':sha256_file(BASE/'model/classifier.joblib'),'S_sha256':sha256_file(S/'final_last.pt'),'F_sha256':sha256_file(F/'final_last.pt'),'F_threshold_sha256':sha256_file(F/'THRESHOLD.json'),'empty_S_policy':'Existing learned S17 then legacy geometric empty fallback unchanged for D-only attribution; log any occurrence and require separate learned-mask resolution before new delivery.'});write_json(RUN/'PREDICTIONS_HASHED_BEFORE_EVAL_GT.json',{'prediction_tree_sha256':sha256_tree(RUN/'predictions'),'baseline':'E17','GT_not_read':True,'n_cases':40,'new_detector_sha256':lock['checkpoint_sha256'],'seconds':time.monotonic()-start});print('E33 PREDICTIONS FROZEN',flush=True)
if __name__=='__main__':main()
