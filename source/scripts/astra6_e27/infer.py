"""Change only C; use predicted-vessel junction features with frozen E17 S and E14 F."""
import json,time
import numpy as np,torch
from scipy.ndimage import map_coordinates
from scripts.astra6_e27.common import RUN,P,anchors,features
from scripts.astra6_e01.e01_common import DATA,TA36_DIR,load_nifti_geometry
from scripts.delivery.geometry import vessel_geometry_fast
import joblib
from scripts.astra6_e14.train import PARENT as FILTER_PARENT
from scripts.astra6_e01.e01_common import load_boxes,select_candidates,nifti_output,sha256_file,sha256_tree,write_json
from scripts.astra6_e04.run_e04 import Segmenter,load_image,crop_volume,PRIOR,largest,fill_ellipsoid
FROZEN=P/'artifacts/astra6_e14_MR_filter_seed_stability_20260909';BASE=P/'artifacts/astra6_e02_multiscale_vessel_signature_20260908T173916Z'
SEG=P/"artifacts/astra6_e17_MR_detector_crop_segmentation_20260909/model/final_last.pt"

def main():
 torch.set_num_threads(1);lock=json.loads((RUN/'model/LOCKED.json').read_text());assert lock['source_gate_passed'];assert sha256_file(RUN/'model/classifier.joblib')==lock['classifier_sha256'];assert sha256_file(SEG)=='ec1d12c8028843d9ea2cc932f4025bf1247c87135bdb0266c1d5cfbfb20fa14d';records=[json.loads(s) for s in (FROZEN/'candidate_predictions.jsonl').read_text().splitlines()];bykey={(r['case_id'],r['original_index']):r for r in records};ids=json.loads((BASE/'eval_case_ids.json').read_text());model=Segmenter().cuda();model.load_state_dict(torch.load(SEG,map_location='cpu',weights_only=False)['state_dict']);model.eval();out=[];start=time.time();classifier=joblib.load(RUN/"model/classifier.joblib");classifier.set_params(n_jobs=1);old=np.load(BASE/"features/eval.npz")["X"];er=json.loads((BASE/"features/eval_records.json").read_text());old_by={(r["case_id"],r["original_index"]):x for r,x in zip(er,old)}
 for n,cid in enumerate(ids,1):
  arr,aff,_=load_image(cid);geom=vessel_geometry_fast(TA36_DIR/f"{cid}.nii.gz",arr.shape,aff);junctions=anchors(geom);mask=np.zeros(arr.shape,np.uint8);bx,sc,_=load_boxes(P/f'artifacts/fold1_eval_center2_epoch60/{cid}_boxes.pkl')
  for idx,score,low,high in reversed(select_candidates(bx,sc)):
   r=bykey[cid,idx];assert np.array_equal(low,r['low']) and np.array_equal(high,r['high']) and score==r['score'];new=dict(r);feat=np.concatenate([old_by[cid,idx],features(junctions,aff,low,high)]);cl=int(classifier.predict(feat[None])[0]);new['baseline_predicted_class_id']=r['predicted_class_id'];new['predicted_class_id']=cl
   if r['filter_keep']:
    x=np.stack([crop_volume(arr,low,high,1),PRIOR])[None].astype(np.float32)
    with torch.inference_mode():prob=model(torch.from_numpy(x).cuda()).sigmoid().cpu().numpy()[0,0]
    lo=np.maximum(np.floor(low).astype(int),0);hi=np.minimum(np.ceil(high).astype(int),arr.shape);grid=np.stack(np.meshgrid(*[np.arange(a,b) for a,b in zip(lo,hi)],indexing='ij'));coords=(grid-((low+high)/2)[:,None,None,None])/(2*np.maximum(high-low,1))[:,None,None,None]*32+15.5;fg=largest(map_coordinates(prob,coords.reshape(3,-1),order=1,mode='constant',cval=0,prefilter=False).reshape(tuple(hi-lo))>=.5);new['empty_fallback']=not fg.any()
    if new['empty_fallback']:fill_ellipsoid(mask,low,high,cl)
    else:mask[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))][fg]=cl
   out.append(new)
  nifti_output(mask,DATA/f'images/{cid}_0000.nii.gz',RUN/f'predictions/mr_center2_k05/{cid}.nii.gz');print('E27',n,40,cid,flush=True)
 (RUN/'candidate_predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in out)+'\n');write_json(RUN/'PREDICTIONS_LOCKED.json',{'prediction_tree_sha256':sha256_tree(RUN/'predictions'),'GT_not_read':True,'n_cases':40,'segmentation_sha256':sha256_file(SEG),'classifier_sha256':lock['classifier_sha256'],'frozen_filter_assignments_sha256':sha256_file(FROZEN/'candidate_predictions.jsonl'),'seconds':time.time()-start})
if __name__=='__main__':main()
