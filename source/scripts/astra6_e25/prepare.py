"""Create inputs with own-case D/S gradient exclusion; legacy GT planning exposure remains."""
import argparse,json,time,re
from pathlib import Path
import numpy as np,torch
from scripts.astra6_e25.common import P,RUN,OOF
from scripts.astra6_e12.common import load_image,crops,nearest_label_identity
from scripts.astra6_e04.run_e04 import Segmenter,PRIOR,SUPPORT
from scripts.astra6_e03.run_e03 import component_records_fast
from scripts.astra6_e01.e01_common import DATA,load_nifti,load_boxes,box_to_native_bounds,select_candidates,sha256_file,write_json

from scripts.astra6_e25.source_candidate_bias import wide_indices,old_source_candidates,summarize as summarize_source_bias

def group(c):return re.sub(r'(_(?:mr|ct)_\d+)_\d+$',r'\1',c)

def prepare(fold):
 torch.set_num_threads(4);split=json.loads((OOF/'source_split.json').read_text())['folds'][fold];ids=split['val'];assert not set(ids)&set(split['train']);lock=json.loads((OOF/f'checkpoints/fold{fold}_OOF_COMPLETE.json').read_text());assert ids==lock['cases'];assert lock['source_split_sha256']==sha256_file(OOF/'source_split.json')
 dev=set(json.loads((P/'artifacts/astra6_e12_MR_expanded_fp_filter_20260909/source_split.json').read_text())['development_cases']);shape=OOF/f'segmenters/fold{fold}/model/final_last.pt';sl=json.loads(shape.with_name('LOCKED.json').read_text());assert sha256_file(shape)==sl['sha256'];sp=shape.parents[1]/'source_split.json';assert sha256_file(sp)==sl['source_split_sha256'];shape_split=json.loads(sp.read_text());shape_train_groups={group(c) for c in shape_split['training_cases']};assert not shape_train_groups&{group(c) for c in ids};assert not {group(c) for c in split['train']}&{group(c) for c in ids};net=Segmenter().cuda().eval();net.load_state_dict(torch.load(shape,map_location='cpu',weights_only=False)['state_dict']);records=[]
 (RUN/'features/cases').mkdir(parents=True,exist_ok=True)
 for n,cid in enumerate(ids,1):
  dest=RUN/f'features/cases/{cid}.npz';meta=dest.with_suffix('.json')
  if dest.exists() and meta.exists() and dest.with_suffix('.provenance.json').exists():
   provenance=json.loads(dest.with_suffix('.provenance.json').read_text());assert provenance['detector_checkpoint_sha256']==lock['checkpoint_sha256'] and provenance['shape_checkpoint_sha256']==sl['sha256'] and provenance['own_case_excluded_from_D_and_S_by_group'] and 'old_source_detector' in provenance;records.extend(json.loads(meta.read_text()));continue
  boxes=OOF/f'oof_boxes/fold{fold}/{cid}_boxes.pkl';assert sha256_file(boxes)==lock['boxes_sha256'][cid];bx,sc,_=load_boxes(boxes);arr,aff,norm=load_image(cid);gt,ga,gs=load_nifti(DATA/f'location_masks/{cid}.nii.gz');assert arr.shape==gt.shape;identity,bound=nearest_label_identity(aff,ga,arr.shape);assert identity,(cid,bound);comps=component_records_fast(gt);deployed={i for i,s,l,h in select_candidates(bx,sc)};xx=[];rr=[];coverage=[]
  candidates=[(i,float(sc[i])) for i in wide_indices(sc)];assert deployed<={i for i,_ in candidates};old_candidates=old_source_candidates(cid,comps)
  for i,score in candidates:
   lo,hi=box_to_native_bounds(bx[i]);counts=[int(np.all((c['coords']>=lo)&(c['coords']<hi),axis=1).sum()) for c in comps];matched=[j for j,(cnt,c) in enumerate(zip(counts,comps)) if cnt/max(1,len(c['coords']))>=.1]
   coverage.append({'candidate_index':i,'score':score,'operating_pool':i in deployed,'components_any_overlap':[j for j,cnt in enumerate(counts) if cnt>0],'components_GT_coverage10pct':matched})
   if not matched and sum(counts)>0:continue
   xx.append(crops(arr,aff,lo,hi));rr.append({'case_id':cid,'fold':fold,'development':cid in dev,'original_index':i,'augmentation':0,'score':score,'low':lo.tolist(),'high':hi.tolist(),'y':int(bool(matched)),'matched_components':matched,'deployed':i in deployed,'case_row':len(xx)-1,'source_detector_and_shape_heldout':True})
  inputs=np.zeros((len(xx),3,32,32,32),np.float16)
  if xx:
   inputs[:,:2]=np.asarray(xx,np.float16)
   with torch.inference_mode():
    for start in range(0,len(inputs),32):
     local=np.asarray(xx[start:start+32],np.float32)[:,0];inp=np.stack([local,np.broadcast_to(PRIOR,local.shape)],axis=1);prob=net(torch.from_numpy(inp).cuda()).sigmoid().cpu().numpy()[:,0]*SUPPORT;inputs[start:start+len(local),2]=prob
  for ri,row in enumerate(rr):
   shaped=inputs[ri,2].astype(np.float32);row.update(predicted_shape_mean_support=float(shaped.sum()/SUPPORT.sum()),predicted_shape_fraction_above_half_support=float((shaped>=.5).sum()/SUPPORT.sum()),box_extent_mm=((np.array(row['high'])-row['low'])*np.linalg.norm(aff[:3,:3],axis=0)).tolist())
  assert np.isfinite(inputs).all();np.savez_compressed(dest,x=inputs);write_json(meta,rr);records.extend(rr);write_json(dest.with_suffix('.provenance.json'),{'detector_checkpoint_sha256':lock['checkpoint_sha256'],'shape_checkpoint_sha256':sl['sha256'],'boxes_sha256':lock['boxes_sha256'][cid],'GT_only_for_candidate_targets':True,'GT_usage_flag_scope':'this feature-preparation stage only; legacy detector planning consumed GT','upstream_supervised_planning_exposure':True,'whole_pipeline_independence_established':False,'planning_audit_sha256':sha256_file(P/'artifacts/research_audit_20260909/PLANNING_PROVENANCE_AUDIT.json'),'shape_training_split_sha256':sl['source_split_sha256'],'own_case_excluded_from_D_and_S_by_group':group(cid) not in shape_train_groups and group(cid) not in {group(c) for c in split['train']},'image_GT_nearest_index_identity':identity,'GT_lesions':[{'component_index':j,'class_id':int(c['class_id']),'voxels':len(c['coords']),'equivalent_diameter_mm':float((6*len(c['coords'])*abs(np.linalg.det(aff[:3,:3]))/np.pi)**(1/3))} for j,c in enumerate(comps)],'candidate_coverage':coverage,'old_source_detector':old_candidates,'coverage_definition':'Any overlap and >=10percent GT-voxel coverage are reported separately; these box diagnostics are not official presence statistics.'});print('E25_OOF_CROPS',fold,n,len(ids),cid,len(rr),flush=True)
 write_json(RUN/f'features/fold{fold}_READY.json',{'cases':ids,'records':len(records),'shape_sha256':sl['sha256'],'OOF_manifest_sha256':sha256_file(OOF/f'checkpoints/fold{fold}_OOF_COMPLETE.json')})

def merge():
 markers=[json.loads((RUN/f'features/fold{i}_READY.json').read_text()) for i in [0,1]];ids=sorted(c for m in markers for c in m['cases']);assert len(ids)==len(set(ids))==267 and not any('center2' in c or '_ct_' in c for c in ids);records=[]
 for cid in ids:records.extend(json.loads((RUN/f'features/cases/{cid}.json').read_text()))
 x=np.lib.format.open_memmap(RUN/'features/images.npy',mode='w+',dtype=np.float16,shape=(len(records),3,32,32,32));offset=0
 for cid in ids:
  z=np.load(RUN/f'features/cases/{cid}.npz')['x'];x[offset:offset+len(z)]=z;offset+=len(z)
 x.flush();assert offset==len(records);(RUN/'features/records.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n');tr=[i for i,r in enumerate(records) if r['fold']==0 and not r['development']];dv=[i for i,r in enumerate(records) if r['fold']==0 and r['development']];assert tr and dv;assert not {group(records[i]['case_id']) for i in tr}&{group(records[i]['case_id']) for i in dv}
 write_json(RUN/'source_split.json',{'training_rows':tr,'development_rows':dv,'final_rows':list(range(len(records))),'development_rule':'Only fold0-heldout cases split by existing F12 source split. Both D/S gradient fitting excludes every development-stage F training and validation MR case; legacy supervised planning does not. Fold1 features not used for development selection, preventing crossfold gradient-fit exposure at this stage, not legacy planning exposure. Final F uses all267OOF cases.','no_MR40_CT5_fit':True,'upstream_plan_TA36_limitations':'Known legacy GT-anchor planning exposure includes MR40 and source OOF cases. TA36 historical exposure also unresolved; no all-stage independent-generalization claim.'});diagnostic=[]
 for cid in ids:
  meta=json.loads((RUN/f'features/cases/{cid}.provenance.json').read_text())
  for lesion in meta['GT_lesions']:
   row={'case_id':cid,**lesion}
   for pool in ['wide','operating']:
    candidates=[c for c in meta['candidate_coverage'] if pool=='wide' or c['operating_pool']]
    for kind in ['any_overlap','GT_coverage10pct']:row[f'{pool}_{kind}']=any(lesion['component_index'] in c[f'components_{kind}'] for c in candidates)
   diagnostic.append(row)
 summary={k:sum(r[k] for r in diagnostic)/len(diagnostic) for k in ['wide_any_overlap','wide_GT_coverage10pct','operating_any_overlap','operating_GT_coverage10pct']}
 write_json(RUN/'evaluation/SOURCE_CANDIDATE_COVERAGE.json',{'GT_lesions':len(diagnostic),'recall':summary,'lesions':diagnostic,'source_OOF':True,'OOF_scope':'own-case gradient exclusion only; known supervised planning exposure','not_official_statistics':True,'possible_later_hypothesis':'If wide-pool recall is inadequate, source-only missed-GT-positive augmentation may be needed to avoid selection bias in the filter; no such augmentation is added in this predeclared E25 experiment.'})
 summarize_source_bias(ids)
 write_json(RUN/'features/READY.json',{'n_cases':267,'rows':len(records),'images_sha256':sha256_file(RUN/'features/images.npy'),'records_sha256':sha256_file(RUN/'features/records.jsonl'),'folds':markers});print('E25_ALL_SOURCE_READY',len(records),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=[0,1]);ap.add_argument('--merge',action='store_true');a=ap.parse_args()
 if a.fold is not None:prepare(a.fold)
 if a.merge:
  merge()
  from scripts.analysis.research_lineage_audit import main as audit_lineage
  audit_lineage()
