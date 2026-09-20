import json,time
import numpy as np,torch
import nibabel as nib
from scipy.ndimage import map_coordinates
from scripts.astra6_e25.common import RUN,Filter
from scripts.astra6_e12.common import crops,load_image
from scripts.astra6_e01.e01_common import P,DATA,BOX_DIR,load_boxes,select_candidates,nifti_output,sha256_file,sha256_tree,write_json
from scripts.astra6_e04.run_e04 import Segmenter,PRIOR,SUPPORT,crop_volume,largest,fill_ellipsoid
BASE=P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z';SHAPE=P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909'

def main():
 torch.set_num_threads(4);lock=json.loads((RUN/'model/LOCKED.json').read_text());protocol=json.loads((RUN/'END_TO_END_PROTOCOL.json').read_text());assert protocol['evaluate_locked_model_even_if_source_gate_fails'];assert lock['model_sha256']==sha256_file(RUN/'model/final_last.pt');assert lock['threshold_sha256']==sha256_file(RUN/'model/THRESHOLD.json');threshold=json.loads((RUN/'model/THRESHOLD.json').read_text())['threshold'];f=Filter().cuda();f.load_state_dict(torch.load(RUN/'model/final_last.pt',map_location='cpu',weights_only=False)['state_dict']);f.eval();s=Segmenter().cuda();s.load_state_dict(torch.load(SHAPE/'model/final_last.pt',map_location='cpu',weights_only=False)['state_dict']);s.eval()
 slock=json.loads((SHAPE/'model/LOCKED.json').read_text());assert sha256_file(SHAPE/'model/final_last.pt')==slock['model_sha256']
 frozen=SHAPE/'candidate_predictions.jsonl';before={(r['case_id'],r['original_index']):r for r in map(json.loads,frozen.read_text().splitlines())};proof={};cache=RUN/'fixed_upstream';cache.mkdir(exist_ok=True)
 ids=json.loads((BASE/'eval_case_ids.json').read_text());rr=[json.loads(z) for z in (BASE/'candidate_predictions.jsonl').read_text().splitlines()];out=[];checks={};dest=RUN/'predictions/mr_center2_k05';dest.mkdir(parents=True,exist_ok=True);start=time.time()
 for n,cid in enumerate(ids,1):
  arr,aff,_=load_image(cid);mask=np.zeros(arr.shape,np.uint8);counterfactual=np.zeros(arr.shape,np.uint8);bx,sc,_=load_boxes(BOX_DIR/f'{cid}_boxes.pkl');sel=select_candidates(bx,sc);probabilities=[];indices=[]
  for rank,(idx,score,lo,hi) in reversed(list(enumerate(sel))):
   r=next(r for r in rr if r['case_id']==cid and r['original_index']==idx);cl=r['predicted_class_id']
   old=before[cid,idx];assert cl==old['predicted_class_id'] and score==old['score'] and np.array_equal(lo,old['low']) and np.array_equal(hi,old['high'])
   image=crops(arr,aff,lo,hi);shape_input=np.stack([image[0],PRIOR])[None].astype(np.float32)
   with torch.inference_mode():
    prob=s(torch.from_numpy(shape_input).cuda()).sigmoid().cpu().numpy()[0,0];features=np.concatenate([image,(prob*SUPPORT)[None]],axis=0);p=float(f(torch.from_numpy(features[None]).cuda()).sigmoid().cpu()[0])
   probabilities.append(prob);indices.append(idx);keep=p>=threshold;fallback=False
   if keep or old['filter_keep']:
    low=np.maximum(np.floor(lo).astype(int),0);high=np.minimum(np.ceil(hi).astype(int),arr.shape);native=np.stack(np.meshgrid(*[np.arange(a,b) for a,b in zip(low,high)],indexing='ij'));coords=(native-((lo+hi)/2)[:,None,None,None])/(2*np.maximum(hi-lo,1))[:,None,None,None]*32+15.5;fg=largest(map_coordinates(prob,coords.reshape(3,-1),order=1,mode='constant',cval=0,prefilter=False).reshape(tuple(high-low))>=.5);fallback=not fg.any()
    for target,retain in [(mask,keep),(counterfactual,old['filter_keep'])]:
     if not retain:continue
     if fallback:fill_ellipsoid(target,lo,hi,cl)
     else:target[tuple(slice(int(a),int(b)) for a,b in zip(low,high))][fg]=cl
   out.append({'case_id':cid,'original_index':idx,'rank':rank,'score':score,'low':lo.tolist(),'high':hi.tolist(),'predicted_class_id':cl,'filter_probability':p,'filter_keep':keep,'predicted_shape_mean_support':float((prob*SUPPORT).sum()/SUPPORT.sum()),'predicted_shape_fraction_above_half_support':float(((prob*SUPPORT)>=.5).sum()/SUPPORT.sum()),'box_extent_mm':((hi-lo)*np.linalg.norm(aff[:3,:3],axis=0)).tolist(),'baseline_filter_keep':old['filter_keep'],'baseline_filter_probability':old.get('filter_probability'),'empty_fallback':bool(fallback and keep)})
  ref=nib.load(str(SHAPE/f'predictions/mr_center2_k05/{cid}.nii.gz'));assert np.allclose(ref.affine,aff,atol=1e-4) and np.array_equal(counterfactual,np.asarray(ref.dataobj)),f'Frozen upstream parity failed: {cid}'
  np.savez_compressed(cache/f'{cid}.npz',original_index=np.asarray(indices),probability=np.asarray(probabilities,dtype=np.float32));proof[cid]={'baseline_native_exact':True,'candidate_count':len(sel),'probabilities_sha256':sha256_file(cache/f'{cid}.npz')};write_json(RUN/'FIXED_UPSTREAM_PARITY.json',{'complete':len(proof)==len(ids),'GT_read':False,'cases':proof,'S_sha256':slock['model_sha256'],'baseline_assignments_sha256':sha256_file(frozen),'protocol_sha256':sha256_file(RUN/'END_TO_END_PROTOCOL.json')})
  nifti_output(mask,DATA/f'images/{cid}_0000.nii.gz',dest/f'{cid}.nii.gz');checks[cid]={'shape':list(arr.shape),'n_selected':len(sel),'n_kept':sum(r['filter_keep'] for r in out if r['case_id']==cid)};print('inference',n,40,cid,checks[cid],flush=True)
 (RUN/'candidate_predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in out)+'\n');write_json(RUN/'prediction_validity.json',checks);write_json(RUN/'PREDICTIONS_HASHED_BEFORE_EVAL_GT.json',{'prediction_tree_sha256':sha256_tree(RUN/'predictions'),'model_sha256':lock['model_sha256'],'threshold':threshold,'n_cases':40,'n_candidates':len(out),'kept':sum(r['filter_keep'] for r in out),'seconds':time.time()-start})
if __name__=='__main__':main()
