from pathlib import Path
import json,time
import numpy as np,torch
from scripts.astra6_e10.prepare import RUN,BASE,SHAPE
from scripts.astra6_e01.e01_common import DATA,BOX_DIR,load_boxes,box_to_native_bounds,nifti_output,sha256_file,sha256_tree,write_json
from scripts.delivery.refine import predict

def main():
 torch.set_num_threads(4);lock=json.loads((RUN/'model/LOCKED.json').read_text());assert sha256_file(RUN/'model/classifier.joblib')==lock['model_sha256'];ids=json.loads((BASE/'eval_case_ids.json').read_text());dest=RUN/'predictions/mr_center2_k05';dest.mkdir(parents=True,exist_ok=True);rows=[];start=time.time()
 for n,cid in enumerate(ids,1):
  image=DATA/f'images/{cid}_0000.nii.gz';mask,ledger=predict(image,BASE.parent/f'ta36_mr_center2_output/{cid}.nii.gz',BOX_DIR/f'{cid}_boxes.pkl',RUN/'model/classifier.joblib',SHAPE/'model/final_last.pt','cuda:0','MR');nifti_output(mask,image,dest/f'{cid}.nii.gz');bx,sc,_=load_boxes(BOX_DIR/f'{cid}_boxes.pkl')
  for r in ledger:
   lo,hi=box_to_native_bounds(bx[r['index']]);rows.append({'case_id':cid,'original_index':r['index'],'score':r['score'],'low':lo.tolist(),'high':hi.tolist(),'predicted_class_id':r['class'],'empty_fallback':r['ellipsoid_empty_fallback']})
  del mask;print('E10 inference',n,40,cid,flush=True)
 assert len(rows)==72;(RUN/'candidate_predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n');write_json(RUN/'PREDICTIONS_HASHED_BEFORE_EVAL_GT.json',{'classifier_sha256':lock['model_sha256'],'prediction_tree_sha256':sha256_tree(RUN/'predictions'),'n_cases':40,'n_candidates':72,'GT_not_read':True,'seconds':time.time()-start})
if __name__=='__main__':main()
