"""Explicit-file K fusion; fixed detector/dense scores, S32 and C02, no GT."""
import argparse,json,time
from pathlib import Path
import numpy as np,nibabel as nib,torch,joblib,SimpleITK as sitk
from scripts.delivery.dense_helpers import normalized_image
from scripts.astra6_e32.train import Model
from scripts.astra6_e32.common import input_crop,native_foreground
from scripts.astra6_e01.e01_common import load_boxes,box_to_native_bounds,nifti_output,compute_feature
from scripts.astra6_e02.run_e02 import multiscale
from scripts.delivery.geometry import vessel_geometry_fast

def main():
 p=argparse.ArgumentParser()
 for k in ['image','predicted-vessel','boxes','dense','classifier','segmentation','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--policy',choices=['dense_fusion','detector_control'],default='dense_fusion');p.add_argument('--detector-threshold',type=float,required=True);p.add_argument('--dense-threshold',type=float,required=True);a=p.parse_args();began=time.monotonic();torch.set_num_threads(1);torch.set_num_interop_threads(1)
 image,aff,norm=normalized_image(a.image);dense=json.loads(a.dense.read_text());assert list(image.shape)==dense['native_shape'] and np.allclose(aff,dense['affine'],atol=1e-4)
 net=Model().cuda().eval();state=torch.load(a.segmentation,map_location='cpu',weights_only=False);assert state['arm']=='normalized';net.load_state_dict(state['state_dict'])
 try:
  geom=vessel_geometry_fast(a.predicted_vessel,image.shape,aff)
 except ValueError as e:
  if 'empty predicted vessel union' in str(e):geom=None
  else:raise
 if geom is None:
  mask=np.zeros(image.shape,np.uint8)
  nifti_output(mask,a.image,a.output);np.savez_compressed(a.output.with_suffix('.npz'))
  a.output.with_suffix('.json').write_text(json.dumps({'candidates':[],'inference_policy':a.policy,'normalization':norm,'seconds':time.monotonic()-began,'empty_vessel':True},indent=2)+'\n')
  return
 classifier=joblib.load(a.classifier);classifier.n_jobs=1
 boxes,scores,_=load_boxes(a.boxes);rows=[];arrays={}
 def refine(low,high):
  crop,origin,step=input_crop(image,np.zeros((1,1,1),np.uint8),low,high,aff,'normalized')
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):logit=net(torch.from_numpy(crop[None].astype(np.float32)).cuda())
  prob=logit.float().sigmoid().cpu().numpy()[0,0];base,fg=native_foreground(prob,origin,step,image.shape);points=np.argwhere(fg)+base
  return np.sort(np.ravel_multi_index(points.T,image.shape)).astype(np.int64) if len(points) else np.empty(0,np.int64)
 def cl(low,high):
  features=np.concatenate([compute_feature(geom,aff,low,high,'MR'),multiscale(geom,aff,low,high)]).astype(np.float32);return int(classifier.predict(features[None])[0])
 mask=np.zeros(image.shape,np.uint8)
 with np.load(a.dense.with_suffix('.npz')) as previous:
  for c in (dense['candidates'] if a.policy=='dense_fusion' else []):
   if c['quality']<a.dense_threshold:continue
   lo=np.asarray(c['low']);hi=np.asarray(c['high']);flat=refine(lo,hi);fallback=not len(flat)
   if fallback:flat=previous['flat_'+str(c['original_index'])]
   label=cl(lo,hi);mask.ravel()[flat]=label;key='dense_'+str(c['original_index']);arrays[key]=flat;rows.append({**c,'branch':'dense','class':label,'voxels':len(flat),'learned_dense_fallback':fallback,'array_key':key})
 selected=[int(i) for i in np.argsort(-scores,kind='stable')[:100] if scores[i]>=.05 and scores[i]>=a.detector_threshold]
 for index in reversed(selected):
  lo,hi=box_to_native_bounds(boxes[index]);flat=refine(lo,hi)
  if not len(flat):continue
  label=cl(lo,hi);mask.ravel()[flat]=label;key='detector_'+str(index);arrays[key]=flat;rows.append({'original_index':index,'score':float(scores[index]),'branch':'detector','class':label,'voxels':len(flat),'array_key':key})
 nifti_output(mask,a.image,a.output);np.savez_compressed(a.output.with_suffix('.npz'),**arrays)
 ref=sitk.ReadImage(str(a.image));out=sitk.ReadImage(str(a.output));corners=np.stack(np.meshgrid(*[[0,n-1] for n in ref.GetSize()],indexing='ij')).reshape(3,-1).T
 error=max(np.linalg.norm(np.asarray(ref.TransformIndexToPhysicalPoint(tuple(map(int,c))))-out.TransformIndexToPhysicalPoint(tuple(map(int,c)))) for c in corners);assert error<1e-4 and np.array_equal(mask,np.asanyarray(nib.load(str(a.output)).dataobj))
 a.output.with_suffix('.json').write_text(json.dumps({'candidates':rows,'inference_policy':a.policy,'normalization':norm,'seconds':time.monotonic()-began,'max_corner_error_mm':float(error),'policy':'dense original order then detector reverse score order; detector overlap priority. NoF14 or S17; emptyS32 dense→learned dense fallback, detector→empty.','no_GT_reads':True},indent=2)+'\n')
if __name__=='__main__':main()
