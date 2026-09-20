"""Audit new S31 case roles against existing actual upstream fit manifests."""
import json
from scripts.astra6_e31.common import P,RUN
from scripts.astra6_e29.prepare import group
from scripts.astra6_e01.e01_common import write_json,sha256_file

def main():
 split=json.loads((RUN/'source_split.json').read_text());rows=[json.loads(s) for s in (RUN/'features/records.jsonl').read_text().splitlines()];dev=set(split['development_cases']);fit={rows[i]['case_id'] for i in split['train_rows']};final={rows[i]['case_id'] for i in split['final_rows']};mr40=set(json.loads((P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z/eval_case_ids.json').read_text()));ct5=set(json.loads((P/'artifacts/astra6_e16_CT_expanded_segmentation_20260909/source_split.json').read_text())['fixed_CT5'])
 assert not {group(c) for c in final}&{group(c) for c in mr40|ct5}
 assert not {group(c) for c in fit}&{group(c) for c in dev}
 oof=P/'artifacts/astra6_e23_MR_oof_candidates_20260909';ds=json.loads((oof/'source_split.json').read_text())['folds'];ss=[json.loads((oof/f'segmenters/fold{f}/source_split.json').read_text()) for f in [0,1]]
 for row in rows:
  if row['kind']=='OOF':
   c=group(row['case_id']);f=row['OOF_fold'];assert c not in {group(x) for x in ds[f]['train']}
   if row['use_train'] or row['use_val']:assert f==1
 for upstream in [ds[1]['train'],ss[1]['training_cases']]:assert not {group(c) for c in upstream}&{group(c) for c in dev}
 case_roles=[]
 for cid in sorted(final):
  rr=[r for r in rows if r['case_id']==cid];case_roles.append({'case_id':cid,'S31_source_gradient_fit':cid in fit,'S31_source_selection':cid in dev,'S31_final_gradient_fit':True,'GT_box_training_rows':sum(r['kind']=='GT' for r in rr),'OOF_rows':sum(r['kind']=='OOF' for r in rr),'OOF_detector_folds':sorted({r['OOF_fold'] for r in rr if r['kind']=='OOF'})})
 write_json(RUN/'CASE_LINEAGE.json',{'cases':case_roles,'source_fit_cases':sorted(fit),'source_selection_cases':sorted(dev),'final_fit_cases':sorted(final),'MR40_CT5_excluded_from_S31_gradients_by_group':True,'source_selection_excluded_from_D1_and_S1_gradients_by_group':True,'source_split_sha256':sha256_file(RUN/'source_split.json'),'organizer_README_sha256':sha256_file(P.parent/'data_topaneu26/README.md'),'organizer_README_vessel_description':'Vessel segmentation masks predicted by TopBrain organizer model.','input_provenance':{'image':'per-image 0.5/99.5 percentiles on nonzero stride-4 image samples for MR and CT; no GT statistics in this S31 stage','candidate_cue':'GT jitter box for supervised training, OOF D1 box for source evaluation, existing deployment D for MR40','vessel':'organizer predicted vessel source bank; TA36 predicted vessel at MR40 inference; training history unknown','old_segmenter':'not a training input; S1 source-validation learned fallback excludes source selection; E17 inference fallback only'},'known_supervised_planning_exposure':True,'whole_pipeline_independence_established':False,'limitations':['Legacy D supervised anchor planning includes MR40 and source validation','Vessel predictor historical training lineage unresolved','Source selection reused from E29 and E30','Source-final S models have different training sizes; source epoch selection is not final fit validation','Patient identity linkage across unrelated case IDs unavailable']})
 print('E31 case lineage passed',len(fit),len(dev),len(final),flush=True)
if __name__=='__main__':main()
