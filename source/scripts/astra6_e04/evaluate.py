from scripts.astra6_e04.run_e04 import *
from scripts.astra6_e03.run_e03 import METRICS,component_records_fast
def evaluate(run):
 from scripts.local_scoring_arena import aggregate,score_case
 from scripts.astra6_e01.evaluate import component_records,rectangle_overlap,ellipsoid_overlap
 lock=read(run/'PREDICTIONS_HASHED_BEFORE_EVAL_GT.json');assert sha256_tree(run/'predictions')==lock['prediction_tree_sha256']
 ids=read(BASE/'eval_case_ids.json'); before=read(BASE/'evaluation/after_per_case.json'); assert [r['case_id'] for r in before]==ids
 after=[]
 for n,cid in enumerate(ids,1):
  after.append({'case_id':cid,'raw':score_case(sitk_array(run/f'predictions/mr_center2_k05/{cid}.nii.gz'),cid)})
  write_json(run/'evaluation/after_partial.json',after);log(f'official scoring {n}/40')
 write_json(run/'evaluation/before_per_case.json',before);write_json(run/'evaluation/after_per_case.json',after)
 aggs={}; cs={}; coverage={}; shapes={}
 for name,pc in [('before',before),('after',after)]:
  ag=aggregate([r['raw'] for r in pc]);aggs[name]=ag
  cs[name]={k:int(sum(ag['per_class'][f'{k}_{i}'] for i in range(1,53))) for k in ('TP','FP','FN')}
  coverage[name]=sum(ag['per_class'][f'TP_{i}']>0 for i in range(1,53))
  vs=[r['raw'][f'DICE_{i}'] for r in pc for i in range(1,53) if r['raw'][f'TP_{i}']>0]
  shapes[name]={'n':len(vs),'mean':float(np.mean(vs)) if vs else None,'median':float(np.median(vs)) if vs else None}
  write_json(run/f'evaluation/{name}_official.json',{**ag,'counts':cs[name],'coverage':coverage[name]})
 delta={k:float(aggs['after']['overall'][k]-aggs['before']['overall'][k]) for k in METRICS}
 cr=rows(run/'candidate_predictions.jsonl'); ledger=[]; totals={'components':0,'rectangle_hit':0,'ellipsoid_hit':0,'unmatched_candidates':0,'before_correct':0,'after_correct':0}
 for cid in ids:
  gt,_,_=load_nifti(DATA/f'location_masks/{cid}.nii.gz');pred,_,_=load_nifti(run/f'predictions/mr_center2_k05/{cid}.nii.gz');box,sc,_=load_boxes(BOX_DIR/f'{cid}_boxes.pkl');sel=select_candidates(box,sc);hit=set()
  for comp in component_records_fast(gt):
   rect=[x for x in sel if rectangle_overlap(comp['coords'],x[2],x[3])]; ell=any(ellipsoid_overlap(comp['coords'],x[2],x[3]) for x in sel);totals['components']+=1;totals['ellipsoid_hit']+=ell
   rr={'case_id':cid,'class_id':comp['class_id'],'component_id':comp['component_id'],'rectangle_hit':bool(rect),'ellipsoid_hit':ell,'before_correct':False,'after_correct':False,'final_correct_overlap':bool(np.any(pred[tuple(comp['coords'].T)]==comp['class_id']))}
   if rect:
    totals['rectangle_hit']+=1;hit.update(x[0] for x in rect);best=sorted(rect,key=lambda x:(-x[1],x[0]))[0];r=next(r for r in cr if r['case_id']==cid and r['original_index']==best[0]);rr.update(selected_index=best[0],before_class=r['before_class'],after_class=r['predicted_class_id'],before_correct=r['before_class']==comp['class_id'],after_correct=r['predicted_class_id']==comp['class_id']);totals['before_correct']+=rr['before_correct'];totals['after_correct']+=rr['after_correct']
   ledger.append(rr)
  totals['unmatched_candidates']+=len(sel)-len(hit)
  del gt,pred
 assert totals['components']==58 and totals['rectangle_hit']==53 and totals['ellipsoid_hit']==53 and totals['unmatched_candidates']==17 and totals['before_correct']==36
 write_json(run/'lesion_ledger.json',ledger);write_json(run/'diagnostics.json',{'totals':totals,'rescued':[r for r in ledger if r['after_correct'] and not r['before_correct']],'lost':[r for r in ledger if r['before_correct'] and not r['after_correct']],'conditional_shape':shapes})
 rng=np.random.default_rng(20260905); samples={k:[] for k in ['MCC','DICE']}
 for b in range(2000):
  ix=rng.integers(0,40,40);ba=aggregate([before[i]['raw'] for i in ix])['overall'];aa=aggregate([after[i]['raw'] for i in ix])['overall']
  for k in samples:samples[k].append(aa[k]-ba[k])
 boot={k:{'n':2000,'seed':20260905,'mean':float(np.mean(v)),'low':float(np.percentile(v,2.5)),'high':float(np.percentile(v,97.5))} for k,v in samples.items()};write_json(run/'paired_bootstrap.json',boot)
 gates={'MCC_nondecrease':delta['MCC']>=-1e-12,'Dice_gain_ge_0.008':delta['DICE']>=.008-1e-12,'assignment_fixed_36':totals['after_correct']==36,'TP_ge_35':cs['after']['TP']>=35,'coverage_ge_11':coverage['after']>=11,'FP_le_33':cs['after']['FP']<=33,'P_R_VS_nondecrease':all(delta[k]>=-1e-12 for k in ['PRECISION','RECALL','VOLSIM']),'HD95_nonincrease':delta['HD95']<=1e-12}
 status='PASS' if all(gates.values()) else 'FAIL'
 result={'gate':status,'checks':gates,'before':aggs['before']['overall'],'after':aggs['after']['overall'],'delta':delta,'counts':cs,'assignment':totals,'coverage':coverage,'bootstrap':boot,'validity':'PASS'}
 write_json(run/'metrics_comparison.json',result);write_json(run/'success_gate.json',{'status':status,'checks':gates})
 write_json(run/'validity_checks.json',{'status':'PASS','frozen_candidates_classes_native_geometry':read(run/'prediction_validity.json'),'source_split':read(run/'source_split.json'),'synthetic_tests':tests(),'all_cases_scored':len(after)==40,'baseline_cached_raw_sha256':sha256_file(BASE/'evaluation/after_per_case.json'),'scorer_hashes':read(run/'config.json')['scorer_hashes'],'prediction_lock':lock})
 (run/'RESULT_TASK2_ASTRA6_E04.md').write_text('# ASTRA6 E04: crop segmentation refinement\n\n'+json.dumps(result,indent=2)+'\n\nSegmentation changed to a box-normalized binary crop U-Net; candidate classes, scores, selection and draw order are frozen to E02. Threshold .5, largest component and empty-mask ellipsoid fallback fixed before inference. Source-only case-grouped development chose duration; final fit uses all source components with jitter and binary mirror augmentation. No center2 refit. Targets and outputs are constrained to the candidate box; source GT-box to detector-box domain gap remains. Ellipsoid-hit diagnostic refers to original frozen ellipsoids, while final_correct_overlap measures refined masks. Center2 is a repeatedly used research holdout, all cases positive; no blind-test or specificity claim. No patient linkage metadata.\n')
 write_json(run/'DONE.json',{'status':'DONE','valid':True,'gate':status,'n_cases':40})
 log(result)
