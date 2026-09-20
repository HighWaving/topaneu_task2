"""Paired end-to-end attribution after locked E25 evaluation; no fitting."""
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from scripts.astra6_e25.common import RUN
from scripts.astra6_e01.e01_common import P,DATA,TA36_DIR,load_nifti,write_json,sha256_tree
from scripts.analysis.paired_segmentation_diagnostics import components
BASE=P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909'
def match(a,b):
 inter=np.array([[len(np.intersect1d(x['flat'],y['flat'],assume_unique=True)) for y in b] for x in a]).reshape(len(a),len(b))
 if not inter.size:return {}
 dice=2*inter/(np.array([len(x['flat']) for x in a])[:,None]+np.array([len(x['flat']) for x in b])[None,:])
 return {int(i):int(j) for i,j in zip(*linear_sum_assignment(-dice)) if inter[i,j]>0}
def describe(comp,aff,vessel):
 pts=comp['coords'];world=pts@aff[:3,:3].T;center=world.mean(0);ev=np.linalg.eigvalsh(np.cov(world.T)) if len(pts)>2 else np.zeros(3);elong=float(np.sqrt(max(ev[-1],0)/max(ev[-2],1e-6)))
 labels,counts=np.unique(vessel[tuple(pts.T)],return_counts=True);hist={str(int(k)):int(v) for k,v in zip(labels,counts)};fraction=1-hist.get('0',0)/len(pts);n_labels=sum(k!='0' and v/len(pts)>=.05 for k,v in hist.items());proxy='multiple_predicted_vessel_labels_contact' if n_labels>=2 else 'elongated_predicted_vessel_aligned' if fraction>=.5 and elong>=3 else 'predicted_vessel_aligned' if fraction>=.5 else 'low_predicted_vessel_overlap'
 return {'predicted_vessel_label_histogram':hist,'predicted_vessel_overlap_fraction':fraction,'anatomical_proxy_type':proxy,'class':comp['class'],'voxels':len(pts),'volume_mm3':float(len(pts)*abs(np.linalg.det(aff[:3,:3]))),'center_world_mm':(center+aff[:3,3]).tolist(),'elongation_proxy':elong,'morphology_proxy':'elongated' if elong>=3 else 'nonelongated','clinical_FP_type':'unassigned_requires_image_review'}
def main():
 proof=json.loads((RUN/'FIXED_UPSTREAM_PARITY.json').read_text());assert proof['complete'] and all(x['baseline_native_exact'] for x in proof['cases'].values())
 locked=json.loads((RUN/'PREDICTIONS_HASHED_BEFORE_EVAL_GT.json').read_text());assert sha256_tree(RUN/'predictions')==locked['prediction_tree_sha256']
 bd=json.loads((BASE/'evaluation/lesion_diagnostics.json').read_text());ad=json.loads((RUN/'evaluation/lesion_diagnostics.json').read_text());candidates=[json.loads(s) for s in (RUN/'candidate_predictions.jsonl').read_text().splitlines()];lesions=[];fp=[];case_rows=[];candidate_diagnostics=[]
 for cid,b in bd['cases'].items():
  a=ad['cases'][cid];assert len(a['lesions'])==len(b['lesions'])
  for i,(old,new) in enumerate(zip(b['lesions'],a['lesions'])):
   assert (old['class'],old['voxels'])==(new['class'],new['voxels']);transition='rescued' if new['matched'] and not old['matched'] else 'lost' if old['matched'] and not new['matched'] else 'common_matched' if new['matched'] else 'common_missed'
   lesions.append({'case_id':cid,'lesion_index':i,'source':'center'+cid.split('_center')[1].split('_')[0],'class':old['class'],'size_bin':old['size_bin'],'transition':transition,'before':old,'after':new})
  gt,aff,_=load_nifti(DATA/f'location_masks/{cid}.nii.gz');old,oa,_=load_nifti(BASE/f'predictions/mr_center2_k05/{cid}.nii.gz');new,na,_=load_nifti(RUN/f'predictions/mr_center2_k05/{cid}.nii.gz');assert np.allclose(aff,oa,atol=1e-4) and np.allclose(aff,na,atol=1e-4)
  vessel,va,_=load_nifti(TA36_DIR/f'{cid}.nii.gz');assert vessel.shape==gt.shape and np.allclose(aff,va,atol=1e-4)
  gc,bc,ac=components(gt),components(old),components(new);mb,ma=match(gc,bc),match(gc,ac);bf=[c for i,c in enumerate(bc) if i not in mb.values()];af=[c for i,c in enumerate(ac) if i not in ma.values()];pairs=match(bf,af)
  assert len(bf)==b['FP'] and len(af)==a['FP']
  for cr in [r for r in candidates if r['case_id']==cid]:
   counts=[int(np.all((c['coords']>=cr['low'])&(c['coords']<cr['high']),axis=1).sum()) for c in gc];positive=[j for j,(cnt,c) in enumerate(zip(counts,gc)) if cnt/len(c['coords'])>=.1];candidate_diagnostics.append({**cr,'GT_positive_by_source_coverage_rule':bool(positive),'ambiguous_partial_GT_overlap':not positive and sum(counts)>0,'GT_components_covered10pct':positive})
  for i,c in enumerate(bf):fp.append({'case_id':cid,'transition':'persistent_FP' if i in pairs else 'removed_FP','before':describe(c,aff,vessel),'after':describe(af[pairs[i]],aff,vessel) if i in pairs else None})
  for j,c in enumerate(af):
   if j not in pairs.values():fp.append({'case_id':cid,'transition':'added_FP','before':None,'after':describe(c,aff,vessel)})
  case_rows.append({'case_id':cid,'FP_before':len(bf),'FP_after':len(af),'removed_FP':len(bf)-len(pairs),'added_FP':len(af)-len(pairs)});print('E25 attribution',cid,flush=True)
 def strata(key):
  out={}
  for k in sorted({str(r[key]) for r in lesions}):
   rs=[r for r in lesions if str(r[key])==k];out[k]={'n':len(rs),**{t:sum(r['transition']==t for r in rs) for t in ['rescued','lost','common_matched','common_missed']},'correct_location_before':sum(r['before'].get('class_correct',False) for r in rs),'correct_location_after':sum(r['after'].get('class_correct',False) for r in rs)}
  return out
 result={'complete':True,'GT_transitions':{t:sum(r['transition']==t for r in lesions) for t in ['rescued','lost','common_matched','common_missed']},'FP_transitions':{t:sum(r['transition']==t for r in fp) for t in ['removed_FP','added_FP','persistent_FP']},'size':strata('size_bin'),'location':strata('class'),'source':strata('source'),'lesions':lesions,'FP_components':fp,'case_counts':case_rows,'changed_candidates':[r for r in candidates if r['filter_keep']!=r['baseline_filter_keep']],'limitations':['FP morphology and predicted-vessel overlap/contact proxies are not adjudicated clinical types; TA36 may miss real vessels','One-to-one overlap pairing can be affected by merges/splits; changed candidate IDs retained for review','MR40 only center2; no multiple-center transfer claim','Official class metrics are separately computed; this lesion matching is diagnostic']}
 source_records=[json.loads(s) for s in (RUN/'features/records.jsonl').read_text().splitlines()]
 def distribution(rows):
  keys=['score','predicted_shape_mean_support','predicted_shape_fraction_above_half_support'];return {'n':len(rows),'quantiles_10_50_90':{key:np.percentile([r[key] for r in rows],[10,50,90]).tolist() if rows else None for key in keys},'box_extent_mm_quantiles_10_50_90':np.percentile([r['box_extent_mm'] for r in rows],[10,50,90],axis=0).tolist() if rows else None}
 input_shift={}
 for positive in [False,True]:
  source=[r for r in source_records if r['deployed'] and bool(r['y'])==positive];deployment=[r for r in candidate_diagnostics if not r['ambiguous_partial_GT_overlap'] and r['GT_positive_by_source_coverage_rule']==positive];input_shift[str(positive)]={'OOF_source_operating':distribution(source),'MR40_operating':distribution(deployment)}
 result['candidate_diagnostics']=candidate_diagnostics;result['input_distribution_shift']=input_shift;result['limitations'].append('Source/deployment input distribution differences mix case domain, D training size, helper-S versus S17 weights and selection. They do not isolate a causal factor.')
 from scripts.analysis.filter_operating_diagnostics import fixed_threshold_metrics
 source_result=json.loads((RUN/'evaluation/SOURCE_RESULT.json').read_text())
 source_operating=[(source_records[i],p,bp) for i,p,bp in zip(source_result['rows'],source_result['new_probabilities'],source_result['baseline_probabilities']) if source_records[i]['deployed']]
 deployment=[r for r in candidate_diagnostics if not r['ambiguous_partial_GT_overlap']]
 threshold=source_result['new_threshold']
 result['fixed_threshold_diagnostics']={
  'source_development_new_F':fixed_threshold_metrics([r['y'] for r,p,bp in source_operating],[p for r,p,bp in source_operating],threshold),
  'source_development_baseline_F':fixed_threshold_metrics([r['y'] for r,p,bp in source_operating],[bp for r,p,bp in source_operating],source_result['baseline_equal_coverage_threshold']),
  'deployment_final_F':fixed_threshold_metrics([r['GT_positive_by_source_coverage_rule'] for r in deployment],[r['filter_probability'] for r in deployment],threshold),
  'interpretation':'Diagnostic only: unchanged source-locked threshold, no MR40 threshold search. Development and final F are different fitted models; threshold transfer assumes comparable probability scale. Source TPR=1 is imposed by calibration, not independent evidence. AUROC/BCE and fixed-threshold errors help distinguish ranking and operating-point failure but case domain and D/S changes remain confounded.'}
 write_json(RUN/'evaluation/PAIRED_ERROR_ATTRIBUTION.json',result)
if __name__=='__main__':main()
