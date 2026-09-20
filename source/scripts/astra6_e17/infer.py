"""Change only S; reuse the locked GT-independent E14 detector/C/F decisions."""
import json,time
import numpy as np,torch
from scipy.ndimage import map_coordinates
from scripts.astra6_e17.prepare import RUN,P,DATA
from scripts.astra6_e14.train import PARENT as FILTER_PARENT
from scripts.astra6_e01.e01_common import load_boxes,select_candidates,nifti_output,sha256_file,sha256_tree,write_json
from scripts.astra6_e04.run_e04 import Segmenter,load_image,crop_volume,PRIOR,largest,fill_ellipsoid
FROZEN=P/'artifacts/astra6_e14_MR_filter_seed_stability_20260909';BASE=P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z'
def main():
 torch.set_num_threads(4);lock=json.loads((RUN/'model/LOCKED.json').read_text());assert sha256_file(RUN/'model/final_last.pt')==lock['model_sha256'] and sha256_file(FROZEN/'candidate_predictions.jsonl')==lock['frozen_filter_assignments_sha256'];records=[json.loads(s) for s in (FROZEN/'candidate_predictions.jsonl').read_text().splitlines()];bykey={(r['case_id'],r['original_index']):r for r in records};ids=json.loads((BASE/'eval_case_ids.json').read_text());model=Segmenter().cuda();model.load_state_dict(torch.load(RUN/'model/final_last.pt',map_location='cpu',weights_only=False)['state_dict']);model.eval();out=[];start=time.time()
 for n,cid in enumerate(ids,1):
  arr,aff,_=load_image(cid);mask=np.zeros(arr.shape,np.uint8);bx,sc,_=load_boxes(P/f'artifacts/fold1_eval_center2_epoch60/{cid}_boxes.pkl')
  for idx,score,low,high in reversed(select_candidates(bx,sc)):
   r=bykey[cid,idx];assert np.array_equal(low,r['low']) and np.array_equal(high,r['high']) and score==r['score'];new=dict(r)
   if r['filter_keep']:
    cl=r['predicted_class_id'];x=np.stack([crop_volume(arr,low,high,1),PRIOR])[None].astype(np.float32)
    with torch.inference_mode():prob=model(torch.from_numpy(x).cuda()).sigmoid().cpu().numpy()[0,0]
    lo=np.maximum(np.floor(low).astype(int),0);hi=np.minimum(np.ceil(high).astype(int),arr.shape);grid=np.stack(np.meshgrid(*[np.arange(a,b) for a,b in zip(lo,hi)],indexing='ij'));coords=(grid-((low+high)/2)[:,None,None,None])/(2*np.maximum(high-low,1))[:,None,None,None]*32+15.5;fg=largest(map_coordinates(prob,coords.reshape(3,-1),order=1,mode='constant',cval=0,prefilter=False).reshape(tuple(hi-lo))>=.5);new['empty_fallback']=not fg.any()
    if new['empty_fallback']:fill_ellipsoid(mask,low,high,cl)
    else:mask[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))][fg]=cl
   out.append(new)
  nifti_output(mask,DATA/f'images/{cid}_0000.nii.gz',RUN/f'predictions/mr_center2_k05/{cid}.nii.gz');print('E17',n,40,cid,flush=True)
 (RUN/'candidate_predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in out)+'\n');write_json(RUN/'PREDICTIONS_LOCKED.json',{'prediction_tree_sha256':sha256_tree(RUN/'predictions'),'GT_not_read':True,'n_cases':40,'segmentation_sha256':lock['model_sha256'],'frozen_filter_assignments_sha256':lock['frozen_filter_assignments_sha256'],'seconds':time.time()-start})
if __name__=='__main__':main()
