"""Frozen detector dense head on explicit image/preprocessing/checkpoint files."""
import argparse,pickle,time,json,hashlib
from pathlib import Path
from functools import partial
import numpy as np,nibabel as nib,torch
from scipy.ndimage import label,find_objects
from omegaconf import OmegaConf
from nndet.inference.loading import load_final_model
from nndet.inference.predictor import Predictor
from nndet.inference.ensembler.segmentation import SegmentationEnsembler
from scripts.delivery.dense_helpers import DenseAdapter

def main():
 p=argparse.ArgumentParser()
 for key in ['image','preprocessed','checkpoint','output']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();began=time.monotonic();torch.set_num_threads(1);torch.set_num_interop_threads(1)
 plan=pickle.loads((a.checkpoint/'plan.pkl').read_bytes());cfg=OmegaConf.load(a.checkpoint/'config.yaml');models=load_final_model(a.checkpoint,cfg,plan,num_models=1,identifier='last');adapter=DenseAdapter(models[0]['model'].model).eval()
 data=np.load(a.preprocessed/'case.npz')['data'];props=pickle.loads((a.preprocessed/'case.pkl').read_bytes());props['transpose_backward']=plan['transpose_backward'];assert data.dtype==np.float32 and np.isfinite(data).all()
 predictor=Predictor(ensembler={'seg':partial(SegmentationEnsembler.from_case,parameters={'use_gaussian':True,'argmax':False})},models=[adapter],crop_size=plan['patch_size'],batch_size=1,overlap=.5,device='cuda:0',ensemble_on_device=False)
 result=predictor.predict_case({'data':data},properties=props,restore=True)['seg'];prob=result['pred_seg'].numpy()[0].transpose(2,1,0);ref=nib.load(str(a.image));assert prob.shape==ref.shape and np.isfinite(prob).all();assert torch.all(predictor.ensembler['seg'].overlap>0)
 cc,count=label(prob>=.5,np.ones((3,3,3),np.uint8));arrays={};rows=[]
 for i,sl in enumerate(find_objects(cc)):
  if sl is None:continue
  local=cc[sl]==i+1;points=np.argwhere(local)+np.asarray([s.start for s in sl]);flat=np.sort(np.ravel_multi_index(points.T,prob.shape)).astype(np.int64);values=prob[sl][local];q=float(values.max());arrays['flat_'+str(i)]=flat
  rows.append({'original_index':i,'quality':q,'score':q,'predicted_voxels':len(flat),'low':points.min(0).tolist(),'high':(points.max(0)+1).tolist()})
 a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output.with_suffix('.npz'),**arrays)
 a.output.write_text(json.dumps({'candidates':rows,'native_shape':list(prob.shape),'affine':ref.affine.tolist(),'preprocessed_array_bytes_sha256':hashlib.sha256(memoryview(np.ascontiguousarray(data)).cast('B')).hexdigest(),'seconds':time.monotonic()-began,'all_preprocessed_voxels_searched':True,'candidate_policy':'native26CC p>=.5; scorefloat32max; no size/count/NMS filter; no GT reads'},indent=2)+'\n')
if __name__=='__main__':main()
