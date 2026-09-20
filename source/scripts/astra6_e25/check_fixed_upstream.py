"""CPU audit of new E25 shared S path against existing MR002 E17 output."""
import os,json,time
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np,torch,nibabel as nib
from scipy.ndimage import map_coordinates
from scripts.astra6_e25.infer import BASE,SHAPE
from scripts.astra6_e12.common import crops,load_image
from scripts.astra6_e01.e01_common import P,BOX_DIR,load_boxes,select_candidates,sha256_file
from scripts.astra6_e04.run_e04 import Segmenter,PRIOR,crop_volume,largest,fill_ellipsoid
from scripts.astra6_e25.attribute import match

def main():
 torch.set_num_threads(1);cid='topaneu_center2_mr_002';start=time.time();arr,aff,_=load_image(cid);model=Segmenter();model.load_state_dict(torch.load(SHAPE/'model/final_last.pt',map_location='cpu',weights_only=False)['state_dict']);model.eval();rows=[json.loads(s) for s in (SHAPE/'candidate_predictions.jsonl').read_text().splitlines()];before={(r['case_id'],r['original_index']):r for r in rows};bx,sc,_=load_boxes(BOX_DIR/f'{cid}_boxes.pkl');mask=np.zeros(arr.shape,np.uint8);checked=0
 for idx,score,lo,hi in reversed(select_candidates(bx,sc)):
  r=before[cid,idx];image=crops(arr,aff,lo,hi);assert np.array_equal(image[0],crop_volume(arr,lo,hi,1));checked+=1
  if not r['filter_keep']:continue
  with torch.inference_mode():prob=model(torch.from_numpy(np.stack([image[0],PRIOR])[None].astype(np.float32))).sigmoid().numpy()[0,0]
  low=np.maximum(np.floor(lo).astype(int),0);high=np.minimum(np.ceil(hi).astype(int),arr.shape);native=np.stack(np.meshgrid(*[np.arange(a,b) for a,b in zip(low,high)],indexing='ij'));coords=(native-((lo+hi)/2)[:,None,None,None])/(2*np.maximum(hi-lo,1))[:,None,None,None]*32+15.5;fg=largest(map_coordinates(prob,coords.reshape(3,-1),order=1,mode='constant',cval=0,prefilter=False).reshape(tuple(high-low))>=.5)
  if fg.any():mask[tuple(slice(int(a),int(b)) for a,b in zip(low,high))][fg]=r['predicted_class_id']
  else:fill_ellipsoid(mask,lo,hi,r['predicted_class_id'])
 ref=np.asanyarray(nib.load(str(SHAPE/f'predictions/mr_center2_k05/{cid}.nii.gz')).dataobj);different=int(np.count_nonzero(mask!=ref))
 # FP matching must count real removed/added components, not only net totals.
 a=[{'flat':np.array([1,2,3])},{'flat':np.array([10,11])}];b=[{'flat':np.array([2,3,4])},{'flat':np.array([20,21])}];assert match(a,b)=={0:0} and match([],b)=={} and match(a,[])=={}
 result={'case_id':cid,'candidate_local_S_inputs_bit_identical':checked,'CPU_S17_native_vs_frozen_GPU_E17_different_voxels':different,'CPU_S17_native_exact':different==0,'matching_added_removed_synthetic_verified':True,'seconds':time.time()-start,'S17_sha256':sha256_file(SHAPE/'model/final_last.pt'),'E25_F_not_trained_or_tested':True,'full40_GPU_counterfactual_parity_still_required':True};dest=P/'artifacts/astra6_e25_oof_shape_fp_filter_20260909/FIXED_UPSTREAM_PREFLIGHT.json';dest.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
