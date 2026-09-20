"""C-only comparison with shared frozen D/S/F and exact native counterfactual."""
import json,time
import numpy as np,torch,nibabel as nib
from scipy.ndimage import map_coordinates
from scripts.astra6_e28.common import P,RUN,BASE
from scripts.astra6_e28.representation import JointAnatomyLocation,resample_ras,shape_to_ras,refine_prediction
from scripts.astra6_e04.run_e04 import load_image,largest,fill_ellipsoid
from scripts.astra6_e01.e01_common import DATA,sha256_file,sha256_tree,write_json,nifti_output
F=P/'artifacts/astra6_e25_oof_shape_fp_filter_20260909'
S=P/'artifacts/astra6_e17_MR_detector_crop_segmentation_20260909'
def draw(mask,prob,lo,hi,cl):
 low=np.maximum(np.floor(lo).astype(int),0);high=np.minimum(np.ceil(hi).astype(int),mask.shape);grid=np.stack(np.meshgrid(*[np.arange(a,b) for a,b in zip(low,high)],indexing='ij'));coords=(grid-((lo+hi)/2)[:,None,None,None])/(2*np.maximum(hi-lo,1))[:,None,None,None]*32+15.5;fg=largest(map_coordinates(prob,coords.reshape(3,-1),order=1,mode='constant',cval=0,prefilter=False).reshape(tuple(high-low))>=.5)
 if fg.any():mask[tuple(slice(int(a),int(b)) for a,b in zip(low,high))][fg]=cl
 else:fill_ellipsoid(mask,lo,hi,cl)
def main():
 torch.set_num_threads(2);lock=json.loads((RUN/'model/LOCKED.json').read_text());assert lock['full_formal_training_complete'] and lock['model_sha256']==sha256_file(RUN/'model/final_last.pt');start=json.loads((RUN/'RESEARCH_START.json').read_text());baseline=F if start['baseline']=='E25' else S;assert sha256_file(baseline/'candidate_predictions.jsonl')==start['baseline_candidates_sha256'];proof=json.loads((F/'FIXED_UPSTREAM_PARITY.json').read_text());assert proof['complete'];model=JointAnatomyLocation().cuda().eval();model.load_state_dict(torch.load(RUN/'model/final_last.pt',map_location='cpu',weights_only=False)['state_dict']);records=[json.loads(s) for s in (baseline/'candidate_predictions.jsonl').read_text().splitlines()];ids=json.loads((BASE/'eval_case_ids.json').read_text());assert len(ids)==40;out=[];checks={};begin=time.monotonic()
 for n,cid in enumerate(ids,1):
  arr,aff,_=load_image(cid);mask=np.zeros(arr.shape,np.uint8);oldmask=np.zeros(arr.shape,np.uint8);cache=F/f'fixed_upstream/{cid}.npz';assert sha256_file(cache)==proof['cases'][cid]['probabilities_sha256'];z=np.load(cache);probabilities={int(i):pr for i,pr in zip(z['original_index'],z['probability'])}
  # Frozen records already contain reversed detector draw order.
  for old in [r for r in records if r['case_id']==cid]:
   row=dict(old);lo,hi=np.array(old['low']),np.array(old['high']);cl=old['predicted_class_id'];prob=probabilities[old['original_index']]
   if old['filter_keep']:
    draw(oldmask,prob,lo,hi,cl)
    if 22<=cl<=35 or 45<=cl<=52:
     center=(lo+hi)/2;local=resample_ras(arr,aff,center,24,48);context=resample_ras(arr,aff,center,80,64);shape=shape_to_ras(prob,lo,hi,aff)
     with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):logits=model(torch.from_numpy(local.astype(np.float32))[None,None].cuda(),torch.from_numpy(context.astype(np.float32))[None,None].cuda(),torch.from_numpy(shape.astype(np.float32))[None,None].cuda(),False)['location_logits']
     pp=logits.float().softmax(1).cpu().numpy()[0];new=refine_prediction(cl,pp);row.update(baseline_class_id=cl,predicted_class_id=new,fine_class_probabilities=pp.tolist());cl=new
    draw(mask,prob,lo,hi,cl)
   out.append(row)
  reference=nib.load(str(baseline/f'predictions/mr_center2_k05/{cid}.nii.gz'));ref=np.asanyarray(reference.dataobj);assert np.allclose(aff,reference.affine,atol=1e-4) and np.array_equal(oldmask,ref),f'Frozen baseline reconstruction mismatch {cid}';assert np.array_equal(mask>0,ref>0),'C-only union geometry changed';nifti_output(mask,DATA/f'images/{cid}_0000.nii.gz',RUN/f'predictions/mr_center2_k05/{cid}.nii.gz');checks[cid]={'baseline_native_exact':True,'foreground_union_exact':True};print('E28 inference',n,40,cid,flush=True)
 (RUN/'candidate_predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in out)+'\n');write_json(RUN/'FIXED_UPSTREAM_PARITY.json',{'baseline':start['baseline'],'cases':checks,'complete':len(checks)==40});write_json(RUN/'PREDICTIONS_HASHED_BEFORE_EVAL_GT.json',{'prediction_tree_sha256':sha256_tree(RUN/'predictions'),'C28_sha256':lock['model_sha256'],'baseline':start['baseline'],'n_cases':40,'seconds':time.monotonic()-begin,'GT_not_read':True})
if __name__=='__main__':main()
