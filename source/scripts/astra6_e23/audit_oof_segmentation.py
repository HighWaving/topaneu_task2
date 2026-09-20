"""Fixed-S source-only paired old/OOF box diagnostic, never training."""
from pathlib import Path
import json,os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np,torch
from scipy.ndimage import map_coordinates
from scripts.astra6_e01.e01_common import P,DATA,load_boxes,select_candidates,write_json,sha256_file
from scripts.astra6_e12.common import nearest_label_identity
from scripts.astra6_e04.run_e04 import Segmenter,load_image,crop_volume,PRIOR,largest,component_records_fast,load_nifti
R=P/'artifacts/astra6_e23_MR_oof_candidates_20260909';OUT=R/'segmentation_source_audit';S=P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909'
def native(prob,lo,hi,shape,points,res=32):
 low=np.maximum(np.floor(lo).astype(int),0);high=np.minimum(np.ceil(hi).astype(int),shape);grid=np.stack(np.meshgrid(*[np.arange(a,b) for a,b in zip(low,high)],indexing='ij'));coords=(grid-((lo+hi)/2)[:,None,None,None])/(2*np.maximum(hi-lo,1))[:,None,None,None]*res+(res-1)/2
 fg=largest(map_coordinates(prob,coords.reshape(3,-1),order=1,mode='constant',cval=0,prefilter=False).reshape(tuple(high-low))>=.5)
 if not fg.any():fg=np.sum(((grid-((lo+hi)/2)[:,None,None,None])/np.maximum((hi-lo)/2,.5)[:,None,None,None])**2,axis=0)<=1
 inside=points[np.all((points>=low)&(points<high),axis=1)];overlap=fg[tuple((inside-low).T)].sum();return {'native_Dice':float(2*overlap/(len(points)+fg.sum())),'relative_volume_error':float(fg.sum()/len(points)-1),'GT_inside_box_fraction':float(len(inside)/len(points))}
def oracle(lo,hi,shape,points,res):
 low=points.min(0)-1;high=points.max(0)+2;small=np.zeros(tuple(high-low),np.float32);small[tuple((points-low).T)]=1;center=(lo+hi)/2;extent=2*np.maximum(hi-lo,1);grid=np.stack(np.meshgrid(*[center[i]+(np.arange(res)-(res-1)/2)/res*extent[i] for i in range(3)],indexing='ij'));prob=map_coordinates(small,(grid-low[:,None,None,None]).reshape(3,-1),order=0,mode='constant',cval=0,prefilter=False).reshape(res,res,res);return native(prob,lo,hi,shape,points,res)
def main():
 OUT.mkdir(exist_ok=True)
 if (OUT/'RESULT.json').exists():return
 torch.set_num_threads(1);locks=[json.loads((R/f'checkpoints/fold{f}_OOF_COMPLETE.json').read_text()) for f in [0,1]];owner={c:f for f,l in enumerate(locks) for c in l['cases']};records=[json.loads(s) for s in (S/'features/train_records.jsonl').read_text().splitlines()];split=json.loads((S/'source_split.json').read_text());groups={}
 for i in split['development_detector_rows']:groups.setdefault(records[i]['case_id'],[]).append((i,records[i]))
 assert not set(groups)&{records[i]['case_id'] for i in split['train_rows']} and all(c in owner and 'center2' not in c for c in groups)
 model=Segmenter();model.load_state_dict(torch.load(S/'model/development_last.pt',map_location='cpu',weights_only=False)['state_dict']);model.eval();rows=[]
 plan={'hypothesis':'In-sample source detector boxes can conceal segmentation fragility to unseen-candidate geometry.','fixed_S_sha256':sha256_file(S/'model/development_last.pt'),'source_split_sha256':sha256_file(S/'source_split.json'),'fit':False,'comparison':'Same source-development GT component and S17 development weights, old versus OOF detector box. D operating pool unchanged .3/top5.','training_trigger':'At least20 unique pairs with >=90% GT coverage in both boxes and mean OOF Dice loss >=.02. Missing/ambiguous/extent-loss cases counted separately.','resolution_diagnostic':'32/48 target round-trip on OOF geometry, not new learned model or threshold sweep','limitations':'Source S17 itself trained on old-D crops; diagnostic only, not whole-pipeline leak-free evaluation. Oracle uses GT for diagnosis only.'};write_json(OUT/'PLAN.json',plan)
 for cid,items in sorted(groups.items()):
  marker=OUT/f'{cid}.json'
  if marker.exists():rows.extend(json.loads(marker.read_text())['rows']);continue
  path=R/f'oof_boxes/fold{owner[cid]}/{cid}_boxes.pkl';assert sha256_file(path)==locks[owner[cid]]['boxes_sha256'][cid];bx,sc,_=load_boxes(path);sel=select_candidates(bx,sc);gt,ga,_=load_nifti(DATA/f'location_masks/{cid}.nii.gz');comps=component_records_fast(gt);image,aff,_=load_image(cid);assert image.shape==gt.shape and nearest_label_identity(aff,ga,image.shape)[0];rr=[]
  for idx,r in items:
   base=records[r['matched_GT_base_row']];comp=next(c for c in comps if c['class_id']==r['source_class_id'] and np.array_equal(c['coords'].min(0)-.5,base['low']) and np.array_equal(c['coords'].max(0)+.5,base['high']));pts=comp['coords'];candidates=[]
   for ci,score,lo,hi in sel:
    cover=np.all((pts>=lo)&(pts<hi),axis=1).mean()
    if cover>=.1:candidates.append((score,ci,lo,hi))
   row={'case_id':cid,'source_row':idx,'class':r['source_class_id'],'GT_voxels':len(pts),'OOF_candidate_found':bool(candidates)}
   if candidates:
    score,ci,lo,hi=max(candidates,key=lambda x:(x[0],-x[1]));row.update(candidate_index=ci,score=score,old_box=[r['low'],r['high']],OOF_box=[lo.tolist(),hi.tolist()]);inputs=np.stack([np.stack([crop_volume(image,*bounds,1),PRIOR]) for bounds in [(np.array(r['low']),np.array(r['high'])),(lo,hi)]]).astype(np.float32)
    with torch.inference_mode():probs=model(torch.from_numpy(inputs)).sigmoid().numpy()[:,0]
    row['old']=native(probs[0],np.array(r['low']),np.array(r['high']),image.shape,pts);row['OOF']=native(probs[1],lo,hi,image.shape,pts);row['oracle32']=oracle(lo,hi,image.shape,pts,32);row['oracle48']=oracle(lo,hi,image.shape,pts,48)
   rr.append(row)
  write_json(marker,{'rows':rr});rows.extend(rr);print('OOF_S_AUDIT',cid,len(rows),31,flush=True)
  del gt,image,comps
 counts={}
 for r in rows:
  if r['OOF_candidate_found']:counts[(r['case_id'],r['candidate_index'])]=counts.get((r['case_id'],r['candidate_index']),0)+1
 paired=[r for r in rows if r['OOF_candidate_found'] and counts[r['case_id'],r['candidate_index']]==1 and min(r['old']['GT_inside_box_fraction'],r['OOF']['GT_inside_box_fraction'])>=.9];delta=float(np.mean([r['OOF']['native_Dice']-r['old']['native_Dice'] for r in paired])) if paired else None
 write_json(OUT/'RESULT.json',{'complete':True,'n_source_lesions':len(rows),'n_unique_extent_qualified_pairs':len(paired),'OOF_minus_old_Dice':delta,'training_trigger_passed':len(paired)>=20 and delta<=-.02,'n_candidate_misses':sum(not r['OOF_candidate_found'] for r in rows),'n_ambiguous_candidate_rows':sum(r['OOF_candidate_found'] and counts[r['case_id'],r['candidate_index']]>1 for r in rows),'all_rows':rows,'paired_source_rows':[r['source_row'] for r in paired],'plan':plan})
if __name__=='__main__':main()
